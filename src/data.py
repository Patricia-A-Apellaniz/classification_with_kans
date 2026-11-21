# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/08/2025

# Package imports
import os

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def scale_numerical_data(data):
    cols_to_scale = [col for col in data.columns if len(data[col].unique()) > 10]
    data_norm = data[cols_to_scale].values
    data_norm = (data_norm - data_norm.mean(axis=0)) / data_norm.std(axis=0)
    data.loc[:, cols_to_scale] = data_norm
    return data


def load_data(
    dataset_name,
    args,
    test_split=0.2,
    n_patients=1000,
    raw=False,
):
    if dataset_name in ['heart', 'diabetes_h', 'diabetes_130']:
        # These data are already scaled and the column names do not have spaces
        data = pd.read_csv(os.path.join(args['data_folder'], f"{dataset_name}_data.csv"))
        # Keep only n_patients
        if not raw:
            data = data.sample(
                n=n_patients,
                random_state=0,
            ).reset_index(drop=True)
        target_name = data.columns[-1]
        x, y = data.drop(columns=[target_name]), data[target_name]

    elif dataset_name == 'obesity' or dataset_name == 'obesity_bin': # See https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
        data = pd.read_csv(os.path.join(args['data_folder'], 'obesity.csv'))

        # Keep only n_patients
        if not raw:
            data = data.sample(
                n=n_patients,
                random_state=0,
            ).reset_index(drop=True)

        # Convert all variables to numeric
        data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
        data['family_history_with_overweight'] = data['family_history_with_overweight'].apply(lambda x: 1 if x == 'yes' else 0)
        data['FAVC'] = data['FAVC'].apply(lambda x: 1 if x == 'yes' else 0)
        data['CAEC'] = data['CAEC'].apply(lambda x: 3 if x == 'Always' else (2 if x == 'Frequently' else (1 if x == 'Sometimes' else 0)))
        data['SMOKE'] = data['SMOKE'].apply(lambda x: 1 if x == 'yes' else 0)
        data['SCC'] = data['SCC'].apply(lambda x: 1 if x == 'yes' else 0)
        data['CALC'] = data['CALC'].apply(lambda x: 3 if x == 'Always' else (2 if x == 'Frequently' else (1 if x == 'Sometimes' else 0)))
        data['MTRANS'] = data['MTRANS'].apply(lambda x: 4 if x == 'Automobile' else (3 if x == 'Motorbike' else (2 if x == 'Bike' else (1 if x == 'Public_Transportation' else 0))))
        data['NObeyesdad'] = data['NObeyesdad'].apply(lambda x: 6 if x == 'Obesity_Type_III' else (5 if x == 'Obesity_Type_II' else (4 if x == 'Obesity_Type_I' else (3 if x == 'Overweight_Level_II' else (2 if x == 'Overweight_Level_I' else (1 if x == 'Normal_Weight' else 0))))))

        if dataset_name == 'obesity_bin':
            data['NObeyesdad'] = data['NObeyesdad'].apply(lambda x: 1 if x > 3 else 0)  # Binary classification

        target_name = 'NObeyesdad'
        x, y = data.drop(columns=[target_name]), data[target_name]

        if not raw:
            x = x.fillna(x.mean())
            x = scale_numerical_data(x)

    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer(as_frame=True)
        x, y = data.data, data.target

        # Rename columns to remove spaces
        new_cols = [col.replace(' ', '_') for col in x.columns]
        x.columns = new_cols
        if not raw:
            x = scale_numerical_data(x)

    else:
        raise ValueError(f"Data name {dataset_name} not found")

    # Classification labels must be numeric, finite integers.
    y = pd.to_numeric(y, errors='raise')

    if y.isna().any() or not np.isfinite(y.to_numpy()).all():
        raise ValueError(
            f'{dataset_name}: target contains missing or infinite values.'
        )

    if not np.equal(y.to_numpy(), np.floor(y.to_numpy())).all():
        raise ValueError(
            f'{dataset_name}: target contains non-integer values.'
        )

    y = y.astype(np.int64)

    if raw:
        return x, y

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=0)

    return x_train, x_test, y_train, y_test


def make_sample_splits(dataset_name, x, y, args):
    """
    Return one fixed test index and nested training indices.

    All configured sizes are constructed before filtering the requested
    sizes, so a size receives the same rows regardless of the selection.
    """
    sizes = args['dataset_sizes'][dataset_name]
    seed = args['split_seed']
    indices = np.arange(len(y))
    labels = np.asarray(y)

    if dataset_name in {'heart', 'diabetes_h', 'diabetes_130'}:
        # Select 12,500 rows, then reserve 2,500 for test.
        selected, _ = train_test_split(
            indices,
            train_size=12500,
            random_state=seed,
            stratify=labels,
        )

        train_pool, test_idx = train_test_split(
            selected,
            test_size=2500,
            random_state=seed,
            stratify=labels[selected],
        )
    else:
        # sklearn determines the integer size of the 20% test split.
        train_pool, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )

    if len(train_pool) != max(sizes):
        raise ValueError(
            f'{dataset_name}: expected {max(sizes)} available training '
            f'rows, found {len(train_pool)}.'
        )

    # Repeatedly select a smaller stratified subset from the previous one.
    nested = {}
    current = train_pool

    for size in sorted(sizes, reverse=True):
        if size < len(current):
            current, _ = train_test_split(
                current,
                train_size=size,
                random_state=seed,
                stratify=labels[current],
            )

        nested[size] = current.copy()

    requested = args['sample_sizes']

    if requested is not None:
        nested = {
            size: nested[size]
            for size in sizes
            if size in requested
        }

    return test_idx, nested


def preprocess_split(x_train, x_test):
    """
    Fit imputation and scaling on x_train and apply them to x_test.

    Columns that are constant in x_train are left unscaled.
    Entirely missing training columns are imputed with zero.
    """
    x_train = x_train.astype(float).copy()
    x_test = x_test.astype(float).copy()

    if np.isinf(x_train.to_numpy()).any() or np.isinf(x_test.to_numpy()).any():
        raise ValueError('Input data contains infinite values.')

    # Means are calculated only from the selected training data.
    imputation_values = x_train.mean().fillna(0.0)

    x_train = x_train.fillna(imputation_values)
    x_test = x_test.fillna(imputation_values)

    means = x_train.mean()
    stds = x_train.std(ddof=0)

    # A constant column is not centered or scaled.
    scalable = stds > 0
    means.loc[~scalable] = 0.0
    stds.loc[~scalable] = 1.0

    x_train = (x_train - means) / stds
    x_test = (x_test - means) / stds

    preprocessing = {
        'columns': x_train.columns.tolist(),
        'imputation_values': imputation_values.to_dict(),
        'means': means.to_dict(),
        'stds': stds.to_dict(),
        'constant_columns': stds.index[~scalable].tolist(),
    }

    return x_train, x_test, preprocessing
