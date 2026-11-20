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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.models.models import Mlp_model, LogisticRegressionModel, RandomForestModel, Kan_model, NAMModel


def expected_calibration_error(y_binary, probabilities, n_bins=10):
    """
    Classwise ECE using equal-width bins on [0, 1].

    y_binary contains 0/1 indicators for one class.
    probabilities contains the predicted probability for that class.
    """
    y_binary = np.asarray(y_binary).reshape(-1)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(y_binary) == 0:
        raise ValueError('ECE requires at least one observation.')

    if len(y_binary) != len(probabilities):
        raise ValueError(
            'ECE labels and probabilities have different lengths.'
        )

    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError('n_bins must be a positive integer.')

    # Values equal to 1 are included in the final bin.
    bin_indices = np.minimum(
        (probabilities * n_bins).astype(int),
        n_bins - 1,
    )

    ece = 0.0

    for bin_index in range(n_bins):
        mask = bin_indices == bin_index

        if not np.any(mask):
            continue

        mean_probability = probabilities[mask].mean()
        observed_frequency = y_binary[mask].mean()

        ece += (
            mask.sum() / len(y_binary)
            * abs(observed_frequency - mean_probability)
        )

    return float(ece)


def get_metrics(
    y_true,
    y_pred,
    y_proba,
    classes=None,
    n_bins=10,
):
    """
    Calculate classification and calibration metrics.

    If y_proba is one-dimensional, it is interpreted as the
    probability of the positive class in a binary problem.

    For a probability matrix, columns must follow the order in classes.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_proba = np.asarray(y_proba, dtype=float)

    if len(y_true) == 0:
        raise ValueError('Metrics require at least one observation.')

    if len(y_pred) != len(y_true):
        raise ValueError('y_true and y_pred have different lengths.')

    if classes is None:
        classes = np.unique(y_true)
    else:
        classes = np.asarray(classes).reshape(-1)

    if len(classes) < 2:
        raise ValueError('Metrics require at least two classes.')

    if not np.array_equal(classes, np.unique(classes)):
        raise ValueError(
            'classes must contain unique labels in sorted order.'
        )

    if not np.array_equal(np.unique(y_true), classes):
        raise ValueError(
            'The classes in y_true do not match the supplied classes.'
        )

    if not np.all(np.isin(y_pred, classes)):
        raise ValueError(
            'y_pred contains labels that are not present in classes.'
        )

    if y_proba.ndim == 1:
        if len(classes) != 2:
            raise ValueError(
                'A one-dimensional probability array requires '
                'a binary problem.'
            )

        y_proba = np.column_stack(
            (1.0 - y_proba, y_proba)
        )

    if y_proba.ndim != 2:
        raise ValueError(
            f'Expected a probability matrix, got shape {y_proba.shape}.'
        )

    if y_proba.shape != (len(y_true), len(classes)):
        raise ValueError(
            'Probability matrix shape does not match '
            'the number of observations and classes.'
        )

    if not np.isfinite(y_proba).all():
        raise ValueError('Probabilities contain NaN or infinite values.')

    if np.any(y_proba < 0) or np.any(y_proba > 1):
        raise ValueError('Probabilities must be between 0 and 1.')

    if not np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError('Each row of probabilities must sum to 1.')

    n_classes = len(classes)

    indicators = (
        y_true[:, None] == classes[None, :]
    ).astype(float)

    classwise_ece = [
        expected_calibration_error(
            indicators[:, class_index],
            y_proba[:, class_index],
            n_bins=n_bins,
        )
        for class_index in range(n_classes)
    ]

    if n_classes == 2:
        positive_y = indicators[:, 1]
        positive_proba = y_proba[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(
                y_true,
                y_pred,
                pos_label=classes[1],
                zero_division=0,
            ),
            'recall': recall_score(
                y_true,
                y_pred,
                pos_label=classes[1],
                zero_division=0,
            ),
            'f1': f1_score(
                y_true,
                y_pred,
                pos_label=classes[1],
                zero_division=0,
            ),
            'roc_auc': roc_auc_score(
                positive_y,
                positive_proba,
            ),
            'ece': classwise_ece[1],
            'brier': np.mean(
                (positive_proba - positive_y) ** 2
            ),
        }

    else:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(
                y_true,
                y_pred,
                average='weighted',
                zero_division=0,
            ),
            'recall': recall_score(
                y_true,
                y_pred,
                average='weighted',
                zero_division=0,
            ),
            'f1': f1_score(
                y_true,
                y_pred,
                average='weighted',
                zero_division=0,
            ),
            'roc_auc': roc_auc_score(
                y_true,
                y_proba,
                labels=classes,
                average='weighted',
                multi_class='ovo',
            ),
            'ece_macro': np.mean(classwise_ece),
            'brier': np.mean(
                np.sum((y_proba - indicators) ** 2, axis=1)
            ),
        }

        precision_by_class, recall_by_class, _, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=classes,
                average=None,
                zero_division=0,
            )
        )

        metrics['f1_macro'] = f1_score(
            y_true,
            y_pred,
            labels=classes,
            average='macro',
            zero_division=0,
        )

        metrics['balanced_accuracy'] = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        for class_index in range(n_classes):
            metrics[f'precision_class_{class_index}'] = (
                precision_by_class[class_index]
            )
            metrics[f'recall_class_{class_index}'] = (
                recall_by_class[class_index]
            )

    # The index in each key corresponds to the position in classes.
    for class_index, value in enumerate(classwise_ece):
        metrics[f'ece_class_{class_index}'] = value

    # Clip only for log loss to avoid log(0).
    epsilon = np.finfo(float).eps
    clipped_proba = np.clip(y_proba, epsilon, 1.0)

    metrics['log_loss'] = -np.mean(
        np.sum(indicators * np.log(clipped_proba), axis=1)
    )

    return {
        name: float(value)
        for name, value in metrics.items()
    }


def get_bootstrap_metrics(
    y_true,
    y_pred,
    y_proba,
    n_bootstrap=1000,
    ci=95,
    seed=0,
    classes=None,
    n_bins=10,
    return_valid_count=False,
):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_proba = np.asarray(y_proba)

    if classes is None:
        classes = np.unique(y_true)
    else:
        classes = np.asarray(classes).reshape(-1)

    if n_bootstrap < 1:
        raise ValueError('n_bootstrap must be at least 1.')

    # Validate the original data and probability/class alignment.
    get_metrics(
        y_true,
        y_pred,
        y_proba,
        classes=classes,
        n_bins=n_bins,
    )

    metrics_list = []
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstrap):
        indices = rng.choice(
            len(y_true),
            len(y_true),
            replace=True,
        )

        y_true_sample = y_true[indices]

        # AUC and the current metric implementation require all classes.
        if not np.array_equal(np.unique(y_true_sample), classes):
            continue

        sample_metrics = get_metrics(
            y_true_sample,
            y_pred[indices],
            y_proba[indices],
            classes=classes,
            n_bins=n_bins,
        )

        metrics_list.append(sample_metrics)

    if not metrics_list:
        raise ValueError(
            'No valid bootstrap samples contained all classes.'
        )

    metric_names = metrics_list[0].keys()

    all_metrics = {
        name: [sample[name] for sample in metrics_list]
        for name in metric_names
    }

    alpha = 1 - ci / 100
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    metrics_with_ci = {}

    for name, values in all_metrics.items():
        values = np.asarray(values)

        lower, upper = np.percentile(
            values,
            [lower_percentile, upper_percentile],
        )

        metrics_with_ci[name] = {
            'mean': float(np.mean(values)),
            f'CI_{ci}%': (float(lower), float(upper)),
        }

    if return_valid_count:
        return metrics_with_ci, len(metrics_list)

    return metrics_with_ci


def add_bootstrap_metrics(
    result,
    n_bootstrap=1000,
    ci=95,
    seed=0,
    n_bins=10,
):
    """
    Calculate bootstrap metrics from stored test predictions.

    The result is updated in memory. No model is trained and
    no file is written by this function.
    """
    required_keys = ('y_test', 'y_pred', 'y_proba', 'classes')

    missing = [
        key
        for key in required_keys
        if key not in result
    ]

    if missing:
        raise ValueError(
            f'Cannot calculate bootstrap metrics. Missing keys: {missing}'
        )

    bootstrap_metrics, n_valid = get_bootstrap_metrics(
        result['y_test'],
        result['y_pred'],
        result['y_proba'],
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        classes=result['classes'],
        n_bins=n_bins,
        return_valid_count=True,
    )

    ci_key = f'CI_{ci}%'

    result['metric_mean'] = {
        metric_name: float(values['mean'])
        for metric_name, values in bootstrap_metrics.items()
    }

    result['metric_ci'] = {
        metric_name: (
            float(values[ci_key][0]),
            float(values[ci_key][1]),
        )
        for metric_name, values in bootstrap_metrics.items()
    }

    result['bootstrap'] = {
        'n_bootstrap': n_bootstrap,
        'n_valid': n_valid,
        'ci': ci,
        'method': 'percentile',
        'seed': seed,
        'ece_bins': n_bins,
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
    nam_multiclass_dataset = (
        model_name == 'nam'
        and dataset_name in {'diabetes_h', 'diabetes_130', 'obesity'}
    )

    if (
        model_name in {'ebm', 'xgboost', 'lightgbm'}
        or nam_multiclass_dataset
    ):
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

    model, hyperparameters = get_model(
        model_name,
        default=False,
    )

    new_models = {'ebm', 'xgboost', 'lightgbm'}
    nam_multiclass = model_name == 'nam' and n_classes > 2

    if model_name in new_models or nam_multiclass:
        cv_inner = StratifiedKFold(
            n_splits=n_splits_cv,
            shuffle=True,
            random_state=0,
        )
    else:
        # Preserve the current search behavior for binary NAM
        # and the original models.
        cv_inner = KFold(
            n_splits=n_splits_cv,
            shuffle=True,
            random_state=0,
        )

    error_score = (
        'raise'
        if model_name in new_models or nam_multiclass
        else 0.0
    )

    if n_classes > 2:
        search = GridSearchCV(
            model,
            hyperparameters,
            scoring=[
                'f1_weighted',
                'roc_auc_ovo_weighted',
                'recall_weighted',
            ],
            refit='f1_weighted',
            cv=cv_inner,
            n_jobs=n_jobs,
            error_score=error_score,
            verbose=4,
        )
    else:
        search = GridSearchCV(
            model,
            hyperparameters,
            scoring=['f1', 'roc_auc', 'recall'],
            refit='f1',
            cv=cv_inner,
            n_jobs=n_jobs,
            error_score=error_score,
            verbose=4,
        )

    start = time.perf_counter()
    result = search.fit(x_train, y_train.squeeze())
    total_search_and_refit_time = time.perf_counter() - start

    training_time = result.refit_time_
    hp_search_time = total_search_and_refit_time - training_time

    return (
        result.best_params_,
        result.best_estimator_,
        hp_search_time,
        training_time,
    )


