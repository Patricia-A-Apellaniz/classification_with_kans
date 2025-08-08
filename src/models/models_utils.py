# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/08/2025

# Package imports
from kan import *
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.models.models import Mlp_model, LogisticRegressionModel, RandomForestModel, Kan_model, NAMModel


# Ancillary methods
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
    else:
        raise ValueError(f"Model name {model_name} not found")


def get_best_params(model_name, x_train, y_train, args):
    n_jobs = args['n_jobs']
    n_splits_cv = args['n_folds']
    n_classes = len(np.unique(y_train))
    if n_classes > 2 and model_name == 'nam':  # NAM does not support multiclass problems
        return None, None
    else:
        model, hyperparameters = get_model(model_name, default=False)

        # Configure the cross-validation procedure for parameter tuning
        cv_inner = KFold(n_splits=n_splits_cv, shuffle=True, random_state=0)

        # Define search
        if n_classes > 2:  # Multiclass problem
            search = GridSearchCV(model,
                                  hyperparameters,
                                  scoring=['f1_weighted', 'roc_auc_ovo_weighted', 'recall_weighted'],
                                  refit='f1_weighted',
                                  cv=cv_inner,
                                  n_jobs=n_jobs,
                                  error_score=0.0,
                                  verbose=4)  # Other option: roc_auc_ovo_weighted
        else:  # Binary problem
            search = GridSearchCV(model,
                                  hyperparameters,
                                  scoring=['f1', 'roc_auc', 'recall'],
                                  refit='f1',
                                  cv=cv_inner,
                                  n_jobs=n_jobs,
                                  error_score=0.0,
                                  verbose=4)

        # Execute search
        result = search.fit(x_train, y_train.squeeze())

        return result.best_params_, result.best_estimator_

