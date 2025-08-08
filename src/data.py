# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/08/2025

# Package imports
import os

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def scale_numerical_data(data):
    cols_to_scale = [col for col in data.columns if len(data[col].unique()) > 10]
    data_norm = data[cols_to_scale].values
    data_norm = (data_norm - data_norm.mean(axis=0)) / data_norm.std(axis=0)
    data.loc[:, cols_to_scale] = data_norm
    return data


def load_data(dataset_name, args, test_split=0.2, n_patients=1000):
    if dataset_name in ['heart', 'diabetes_h', 'diabetes_130']:
        # These data are already scaled and the column names do not have spaces
        data = pd.read_csv(os.path.join(args['data_folder'], f"{dataset_name}_data.csv"))
        # Keep only n_patients
        data = data.sample(n=n_patients, random_state=0).reset_index(drop=True)
        target_name = data.columns[-1]
        x, y = data.drop(columns=[target_name]), data[target_name]

    elif dataset_name == 'obesity' or dataset_name == 'obesity_bin': # See https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
        data = pd.read_csv(os.path.join(args['data_folder'], 'obesity.csv'))

        # Keep only n_patients
        data = data.sample(n=n_patients, random_state=0).reset_index(drop=True)

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

        # Impute missing values
        data = data.fillna(data.mean())
        target_name = 'NObeyesdad'
        x, y = data.drop(columns=[target_name]), data[target_name]
        x = scale_numerical_data(x)

    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer(as_frame=True)
        x, y = data.data, data.target

        # Rename columns to remove spaces
        new_cols = [col.replace(' ', '_') for col in x.columns]
        x.columns = new_cols
        x = scale_numerical_data(x)

    else:
        raise ValueError(f"Data name {dataset_name} not found")

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=0)

    return x_train, x_test, y_train, y_test