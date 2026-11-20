# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/08/2025

# Package imports
import os
import pickle
import time
import json

from kan import *
from scipy.stats import t
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.models.models import Mlp_model, LogisticRegressionModel, RandomForestModel, Kan_model, NAMModel


def get_metrics(y_true, y_pred, y_proba):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    y_proba = np.squeeze(y_proba)

    binary = False
    if len(y_proba.shape) == 1:
        binary = True
    elif y_proba.shape[1] <= 2:
        binary = True
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]

    # Check for the right method to apply
    if binary:  # Binary classification
        return {'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_proba)}
    else:  # Multiclass classification
        return {'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovo')}


def get_bootstrap_metrics(y_true, y_pred, y_proba, n_bootstrap=1000, ci=95, seed=0):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    original_classes = np.unique(y_true)
    metrics_list = []
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        y_true_sample = y_true[indices]

        # Check if all classes are present in the sample
        if not np.all(np.isin(original_classes, np.unique(y_true_sample))):
            continue  # Skip this iteration if not all classes are present

        sample_metrics = get_metrics(y_true_sample, y_pred[indices], y_proba[indices])
        metrics_list.append(sample_metrics)

    # Group metrics by name
    metric_names = metrics_list[0].keys()
    all_metrics = {k: [] for k in metric_names}
    for m in metrics_list:
        for k in m:
            all_metrics[k].append(m[k])

    # # Calculate mean and confidence intervals
    # alpha = 1 - ci / 100
    # t_val = t.ppf(1 - alpha / 2, df=n_bootstrap - 1)
    # metrics_with_ci = {}
    #
    # for k, values in all_metrics.items():
    #     values = np.array(values)
    #     mean = np.mean(values)
    #     std_err = np.std(values, ddof=1) / np.sqrt(n_bootstrap)
    #     ci_range = t_val * std_err
    #     metrics_with_ci[k] = {
    #         'mean': mean,
    #         f'CI_{ci}%': (mean - ci_range, mean + ci_range)
    #     }
    #
    # return metrics_with_ci
    # Calculate the bootstrap mean and percentile confidence interval
    alpha = 1 - ci / 100
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    metrics_with_ci = {}

    for k, values in all_metrics.items():
        values = np.asarray(values)
        mean = np.mean(values)
        lower, upper = np.percentile(
            values, [lower_percentile, upper_percentile]
        )

        metrics_with_ci[k] = {
            'mean': mean,
            f'CI_{ci}%': (lower, upper)
        }

    return metrics_with_ci


def add_bootstrap_metrics(result, n_bootstrap=1000, ci=95, seed=0):
    """
    Calculate bootstrap metrics from stored test predictions.

    The result dictionary is updated in memory.
    This function does not train a model or write to disk.
    """
    required_keys = ('y_test', 'y_pred', 'y_proba')

    missing = [key for key in required_keys if key not in result]

    if missing:
        raise ValueError(
            f"Cannot calculate bootstrap metrics. Missing keys: {missing}"
        )

    bootstrap_metrics = get_bootstrap_metrics(
        result['y_test'], result['y_pred'], result['y_proba'],
        n_bootstrap=n_bootstrap, ci=ci, seed=seed,
    )
    ci_key = f'CI_{ci}%'
    result['metric_mean'] = {
        metric_name: float(values['mean'])
        for metric_name, values in bootstrap_metrics.items()
    }
    result['metric_ci'] = {
        metric_name: (float(values[ci_key][0]), float(values[ci_key][1]))
        for metric_name, values in bootstrap_metrics.items()
    }
    result['bootstrap'] = {
        'n_bootstrap': n_bootstrap,
        'ci': ci,
        'method': 'percentile',
        'seed': seed,
    }
    return result


def get_params(model_name, default):
    if model_name == 'mlp':
        # MLP model parameters
        params = {'hidden_layer_sizes': [(32,), (64,), (128,), (256,)],  # the number of neurons in the hidden layers
                  'max_iter': [10000],  # the maximum number of iterations
                  'early_stopping': [True],
                  # whether to use early stopping to terminate training when validation score is not improving
                  'alpha': [0.0001, 0.001],  # L2 penalty (regularization term) parameter
                  }

    elif model_name == 'lr':
        # LR model parameters
        params = {'C': [0.1],  # regularization strength; smaller values specify stronger regularization
                  'penalty': ['l2', 'l1'],  # type of regularization to use ('l1', 'l2', or none)
                  'solver': ['liblinear'],  # optimization solvers
                  'max_iter': [1000],  # maximum number of iterations for solvers
                  'class_weight': ['balanced', None],  # adjust weights inversely proportional to class frequencies
                  'random_state': [0],  # the seed used by the random number generator
                  }

    elif model_name == 'rf':
        # RF model parameters
        params = {'n_estimators': [20, 50],  # the number of trees in the forest
                  'criterion': ['gini'],  # the function to measure the quality of a split (default='gini')
                  'max_depth': [10, 20],  # the maximum depth of the tree
                  'min_samples_split': [2],  # the minimum number of samples required to split an internal node
                  'min_samples_leaf': [1, 5],  # the minimum number of samples required to be at a leaf node
                  'class_weight': ['balanced', None],
                  'max_features': ['log2'],  # the number of features to consider when looking for the best split
                  'bootstrap': [True],  # whether bootstrap samples are used when building trees (default=True)
                  'random_state': [0],  # the seed used by the random number generator (default=0)
                  'n_jobs': [1],  # the number of jobs to run in parallel for both fit and predict (default=5)
                  }

    elif model_name in ['kan', 'kan_gam']:
        # KAN model parameters
        params = {'hidden_dim': [0, 5, [5, 5]],  # the dimension of the hidden layers
                  'batch_size': [-1],  # the number of samples to use for each training step (i.e., use all of them)
                  'grid': [1, 3, 5],  # the number of grid points in the input space
                  'k': [1, 3, 5],  # the polynomial order in the spline
                  'seed': [0],  # the seed used by the random number generator
                  'lr': [0.001],  # the learning rate
                  'early_stop': [True],
                  # whether to use early stopping to terminate training when validation score is not improving
                  'steps': [10000],  # the number of training steps
                  'lamb': [0.1, 0.01, 0.001],  # the regularization strength
                  'lamb_entropy': [0.1],  # the regularization strength for the entropy term
                  'weight': [True, False],  # whether to use the weight term (i.e., to balance the classes)
                  'sparse_init': [True, False],  # whether to use a sparse initialization
                  'mult_kan': [False],  # whether to use multiplication nodes in the KAN model
                  }

        if model_name == 'kan_gam':
            params['hidden_dim'] = [0]  # The hidden dimension is not used in the GAM version
            params['mult_kan'] = [False]  # The GAM version does not use multiplication nodes

    elif model_name == 'nam':
        # NAM model parameters
        params = {'num_epochs': [1000],
                  'num_learners': [10, 20],
                  'metric': ['aucroc'],
                  'early_stop_mode': ['max'],
                  'n_jobs': [1],
                  'random_state': [0],
                  'num_basis_functions': [32, 64, 128],
                  'hidden_size': [[64, 32], [128, 64]],
                  }
    elif model_name == 'ebm':
        params = {
            'interactions': [0, 5],
            'learning_rate': [0.01, 0.05],
            'max_leaves': [2, 3],
            'max_rounds': [500],
            'outer_bags': [4],
            'random_state': [0],
            'n_jobs': [1],
        }

    elif model_name == 'xgboost':
        params = {
            'n_estimators': [100, 300],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1],
            'random_state': [0],
            'n_jobs': [1],
        }

    elif model_name == 'lightgbm':
        params = {
            'n_estimators': [100, 300],
            'num_leaves': [15, 31],
            'learning_rate': [0.05, 0.1],
            'random_state': [0],
            'n_jobs': [1],
            'verbosity': [-1],
        }
    else:
        raise ValueError(f"Model name {model_name} not recognized")

    # If default, select the first value of each parameter
    if default:
        for key in params.keys():
            params[key] = params[key][0]

    return params


def get_model(model_name, default=False):
    params = get_params(model_name, default)
    if model_name == 'mlp':
        return Mlp_model(), params
    elif model_name == 'lr':
        return LogisticRegressionModel(), params
    elif model_name == 'rf':
        return RandomForestModel(), params
    elif model_name == 'kan' or model_name == 'kan_gam':
        return Kan_model(), params
    elif model_name == 'nam':
        return NAMModel(), params
    elif model_name == 'ebm':
        return ExplainableBoostingClassifier(), params
    elif model_name == 'xgboost':
        return XGBClassifier(), params
    elif model_name == 'lightgbm':
        return LGBMClassifier(), params
    else:
        raise ValueError(f"Model name {model_name} not found")


def load_saved_params(dataset_name, model_name, args):
    """
    Load previously selected hyperparameters for one dataset and model.

    Supports historical top-level parameters and the new nested format.
    Only load pickle files from trusted sources.
    """
    result_path = os.path.join(
        args['params_folder'], dataset_name, model_name + '.pkl',
    )
    if not os.path.isfile(result_path):
        raise FileNotFoundError(
            f"Saved hyperparameters were not found: {result_path}"
        )
    with open(result_path, 'rb') as f:
        saved_result = pickle.load(f)
    if not isinstance(saved_result, dict):
        raise ValueError(f"Expected a dictionary in {result_path}.")

    # The current model's parameter grid defines the accepted keys.
    parameter_names = get_params(model_name, default=False).keys()
    if 'hyperparameters' in saved_result:
        source = saved_result['hyperparameters']
    else:
        source = saved_result
    missing = [name for name in parameter_names if name not in source]
    if missing:
        raise ValueError(
            f"Missing hyperparameters in {result_path}: {missing}"
        )
    params = {name: source[name] for name in parameter_names}

    # JSON configurations introduced later may represent tuples as lists.
    if model_name == 'mlp':
        params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
    return params


def load_configured_params(dataset_name, model_name, train_size, args):
    if model_name in {'ebm', 'xgboost', 'lightgbm'}:
        path = os.path.join(
            args['config_folder'],
            dataset_name,
            model_name,
            f'{train_size}.json',
        )

        with open(path, encoding='utf-8') as f:
            params = json.load(f)

    else:
        path = os.path.join(
            args['config_folder'],
            dataset_name,
            model_name + '.json',
        )

        with open(path, encoding='utf-8') as f:
            configurations = json.load(f)

        params = configurations[str(train_size)]

        if model_name == 'mlp':
            params['hidden_layer_sizes'] = tuple(
                params['hidden_layer_sizes']
            )

    return params


def fit_with_saved_params(
    model_name,
    dataset_name,
    x_train,
    y_train,
    args,
    train_size,
):
    """Train one model with saved parameters, without GridSearchCV."""
    if model_name == 'nam' and len(np.unique(y_train)) > 2:
        return None, None, None, None

    params = load_configured_params(
        dataset_name,
        model_name,
        train_size,
        args,
    )

    model, _ = get_model(model_name, default=False)
    model.set_params(**params)

    start = time.perf_counter()
    model.fit(x_train, np.asarray(y_train).ravel())
    training_time = time.perf_counter() - start

    hp_search_time = None

    return params, model, hp_search_time, training_time


def get_best_params(model_name, x_train, y_train, args):
    n_jobs = args['n_jobs']
    n_splits_cv = args['n_folds']
    n_classes = len(np.unique(y_train))
    if n_classes > 2 and model_name == 'nam':  # NAM does not support multiclass problems
        return None, None, None, None
    else:
        model, hyperparameters = get_model(model_name, default=False)

        # Configure the cross-validation procedure for parameter tuning
        if model_name in {'ebm', 'xgboost', 'lightgbm'}:
            cv_inner = StratifiedKFold(
                n_splits=n_splits_cv,
                shuffle=True,
                random_state=0,
            )
        else:
            cv_inner = KFold(
                n_splits=n_splits_cv,
                shuffle=True,
                random_state=0,
            )

        error_score = (
            'raise'
            if model_name in {'ebm', 'xgboost', 'lightgbm'}
            else 0.0
        )

        # Define search
        if n_classes > 2:  # Multiclass problem
            search = GridSearchCV(model,
                                  hyperparameters,
                                  scoring=['f1_weighted', 'roc_auc_ovo_weighted', 'recall_weighted'],
                                  refit='f1_weighted',
                                  cv=cv_inner,
                                  n_jobs=n_jobs,
                                  error_score=error_score,
                                  verbose=4)  # Other option: roc_auc_ovo_weighted
        else:  # Binary problem
            search = GridSearchCV(model,
                                  hyperparameters,
                                  scoring=['f1', 'roc_auc', 'recall'],
                                  refit='f1',
                                  cv=cv_inner,
                                  n_jobs=n_jobs,
                                  error_score=error_score,
                                  verbose=4)

        # Search includes cross-validation and refitting the best model.
        start = time.perf_counter()
        result = search.fit(x_train, y_train.squeeze())
        total_search_and_refit_time = time.perf_counter() - start

        # GridSearchCV records the time used to refit the best estimator.
        training_time = result.refit_time_

        # Search and cross-validation time, excluding the final refit.
        hp_search_time = (
            total_search_and_refit_time - training_time
        )

        return (
            result.best_params_,
            result.best_estimator_,
            hp_search_time,
            training_time,
        )


