# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 31/07/2025

# Package imports
import os
import sys
import time
import pickle

import numpy as np

from tabulate import tabulate

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from src.data import load_data
from src.models.model_utils import get_best_params, get_bootstrap_metrics
from src.utils import get_config, get_results_table, get_p_values_from_table_data, create_results_folder


def get_mrr_datasets(mrr_all_elements):
    for model in args['models']:
        if len(mrr_all_elements[model]) > 0:
            mrr_all_elements[model] = np.round(sum([1 / v for v in mrr_all_elements[model]]) / len(mrr_all_elements[model]), 2)
        else:
            mrr_all_elements[model] = 0

    # Sort MRR in descending order
    mrr_all_elements = {k: v for k, v in sorted(mrr_all_elements.items(), key=lambda item: item[1], reverse=True)}
    print(f"\n\n---------------MRR for all metrics---------------")
    for model in mrr_all_elements.keys():
        print(f"{model}: {mrr_all_elements[model]:.2f}")


def get_p_values():
    print("\n\n\n-----------P-values computation for all metrics and models-----------")
    # Get p-values too (to compare them with MRR)
    data_list = []
    models_without_nam = args['models'].copy()
    if 'nam' in models_without_nam:
        models_without_nam.remove('nam')
    for model in models_without_nam:
        model_data = []
        for metric_idx in range(2, len(table_col_names) - 1):  # Skip 'Dataset' and 'Model' columns, and 'Time' column
            metric_data = []
            for dataset in args['datasets']:
                # Find the corresponding value in results_table_no_ci
                for row in results_table_no_ci:
                    if row[0] == dataset and row[1] == model:
                        metric_data.append(row[metric_idx])
                        break
            model_data.append(metric_data)
        data_list.append(model_data)

    # If value of dimensions is not consistent, pad with np.nan
    data_list = [np.array(d) for d in data_list]
    max_datasets = max([d.shape[1] for d in data_list])
    for i in range(len(data_list)):
        if data_list[i].shape[1] < max_datasets:
            pad_width = max_datasets - data_list[i].shape[1]
            data_list[i] = np.pad(data_list[i], ((0, 0), (0, pad_width)), mode='constant', constant_values=np.nan)
    data = np.array(data_list)  # Shape: (num_models, num_metrics, num_datasets)
    get_p_values_from_table_data(data, list_of_methods=args['models'], list_of_metrics=table_col_names[2:-1])


def get_mrr_models():
    mrr_table_col_names = ['Metric name'] + args['models']
    mrr_results_table = []
    mrr_all_elements = {model_name: [] for model_name in args['models']}
    for idx, metric_name in enumerate(table_col_names[2:-1]):  # Skip the first two columns (Dataset and Model)
        mrr = {model_name: [] for model_name in args['models']}

        for dataset in args['datasets']:
            values = [{res[1]: float(res[idx+2].split(' ')[0])} for res in results_table if res[0] == dataset and float(res[idx+2].split(' ')[0]) > -0.5]  # Note htat -0.5 is just a threshold: we put -1 to flag the metrics that were not computed
            values = sorted(values, key=lambda x: list(x.values())[0], reverse=True)
            val = 1  # Initial value for the rank
            for j in range(len(values)):
                if j > 0:
                    if abs(list(values[j].values())[0] - list(values[j - 1].values())[0]) > 0.001:
                        val += 1  # Increase the rank only if this value is (too) different from the previous one!
                mrr[list(values[j].keys())[0]].append(val)
                mrr_all_elements[list(values[j].keys())[0]].append(val)
        # print(f"metric_name: {metric_name}, mrr: {mrr}")
        for key in mrr.keys():
            if len(mrr[key]) > 0:
                mrr[key] = sum([1 / v for v in mrr[key]]) / len(mrr[key])
            else:
                mrr[key] = 0
        mrr_results_table.append([metric_name])
        for model in args['models']:
            mrr_results_table[-1].extend([mrr[model]])

    print('\n\n---------------MRR for each metric among all models---------------\n')
    # print(tabulate(mrr_results_table, headers=mrr_table_col_names, tablefmt='latex', floatfmt=".2f"))
    print(tabulate(mrr_results_table, headers=mrr_table_col_names, floatfmt=".2f"))

    return mrr_all_elements


if __name__ == '__main__':
    # Get the configuration
    args = get_config('performance')
    create_results_folder(args['results_folder'], args)

    if args['train']:
        for dataset_name in args['datasets']:
            # Load data
            x_train, x_test, y_train, y_test = load_data(dataset_name, args)

            for model_name in args['models']:
                print(f"\n\nTraining {model_name}")
                t0 = time.time()
                best_params, best_model = get_best_params(model_name, x_train, y_train, args)
                train_time = time.time() - t0

                if best_model is None:
                    metrics = get_bootstrap_metrics(y_test,
                                                    y_test,
                                                    np.ones((y_test.shape[0], len(np.unique(y_train)))) / len(
                                                        np.unique(y_train)))

                    # Set all metrics to -1 (flag value)
                    metrics = {key: -1 for key in metrics.keys()}
                    metrics['time'] = train_time

                else:
                    y_pred = best_model.predict(x_test)
                    y_proba = best_model.predict_proba(x_test)
                    # metrics = get_metrics(y_test, y_pred, y_proba)
                    metrics = get_bootstrap_metrics(y_test, y_pred, y_proba)
                    print(f"{model_name} trained in {train_time:.4f} seconds. Metrics:")
                    print(metrics)
                    metrics.update(best_params)
                    metrics['dataset'] = dataset_name
                    metrics['time'] = train_time

                # Save the metrics
                with open(os.path.join(args['results_folder'], dataset_name, model_name + '.pkl'), 'wb') as f:
                    pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    #### Show results
    results_table, table_col_names, results_table_no_ci = get_results_table(args)
    mrr_all = get_mrr_models()  # Compute MRR for each metric among all models
    get_mrr_datasets(mrr_all)  # Compute MRR for each model among all datasets
    get_p_values()  # Get p-values to compare them with MRR values
