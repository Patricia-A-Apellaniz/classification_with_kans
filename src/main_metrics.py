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
from src.utils import get_config
from src.models.models_utils import get_best_params, get_metrics, get_model


def get_results_table():
    table_col_names = ['Dataset', 'Model', 'Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Time']
    results_table = []
    for dataset in args['datasets']:
        for model_name in args['models']:
            # Load the metrics
            with open(os.path.join(args['results_folder'], dataset, model_name + '.pkl'), 'rb') as f:
                metrics = pickle.load(f)

            # Approximate all metrics to two decimal places
            for key in ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']:
                metrics[key] = np.round(metrics[key], 2)

            # Add the avg time to the results
            _, p = get_model(model_name)
            l = [len(p[key]) for key in p.keys()]
            avg_time = metrics['time'] / np.prod(l)
            # print(f"Dataset: {dataset}, model: {model_name}, metrics: {metrics}")
            results_table.append([dataset,
                                  model_name,
                                  metrics['accuracy'],
                                  metrics['roc_auc'],
                                  metrics['f1'],
                                  metrics['precision'],
                                  metrics['recall'],
                                  avg_time])
    print('\n\nResults table:\n')
    print(tabulate(results_table, headers=table_col_names, tablefmt='latex', floatfmt=".2f"))

    return results_table, table_col_names


def get_mrr_models(results_table, table_col_names, args):
    mrr_table_col_names = ['Metric name'] + args['models']
    mrr_results_table = []
    mrr_all = {model_name: [] for model_name in args['models']}
    for idx, metric_name in enumerate(table_col_names[2:]):  # Skip the first two columns (Dataset and Model)
        mrr = {model_name: [] for model_name in args['models']}

        for dataset in args['datasets']:
            values = [{res[1]: res[idx + 2]} for res in results_table if res[0] == dataset and res[
                idx + 2] > -0.5]  # Note htat -0.5 is just a threshold: we put -1 to flag the metrics that were not computed
            values = sorted(values, key=lambda x: list(x.values())[0], reverse=True)
            val = 1  # Initial value for the rank
            for j in range(len(values)):
                if j > 0:
                    if abs(list(values[j].values())[0] - list(values[j - 1].values())[0]) > 0.001:
                        val += 1  # Increase the rank only if this value is (too) different from the previous one!
                mrr[list(values[j].keys())[0]].append(val)
                mrr_all[list(values[j].keys())[0]].append(val)
        # print(f"metric_name: {metric_name}, mrr: {mrr}")
        for key in mrr.keys():
            if len(mrr[key]) > 0:
                mrr[key] = sum([1 / v for v in mrr[key]]) / len(mrr[key])
            else:
                mrr[key] = 0
        mrr_results_table.append([metric_name])
        for model in args['models']:
            mrr_results_table[-1].extend([mrr[model]])

    print('\n\nMRR for each metric among all models:\n')
    print(tabulate(mrr_results_table, headers=mrr_table_col_names, tablefmt='latex', floatfmt=".2f"))

    return mrr_all


def get_mrr_datasets(mrr_all, args):
    for model_name in args['models']:
        if len(mrr_all[model_name]) > 0:
            mrr_all[model_name] = np.round(sum([1 / v for v in mrr_all[model_name]]) / len(mrr_all[model_name]), 2)
        else:
            mrr_all[model_name] = 0

    # Sort MRR in descending order
    mrr_all = {k: v for k, v in sorted(mrr_all.items(), key=lambda item: item[1], reverse=True)}
    print(f"\n\nMRR for all metrics: {mrr_all}")


if __name__ == '__main__':
    # Get the configuration
    args = get_config('metrics')

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
                    metrics = get_metrics(y_test,
                                          y_test,
                                          np.ones((y_test.shape[0], len(np.unique(y_train)))) / len(np.unique(y_train)))

                    # Set all metrics to -1 (flag value)
                    metrics = {key: -1 for key in metrics.keys()}
                    metrics['time'] = train_time

                else:
                    y_pred = best_model.predict(x_test)
                    y_proba = best_model.predict_proba(x_test)
                    metrics = get_metrics(y_test, y_pred, y_proba)
                    print(f"{model_name} trained in {train_time:.4f} seconds. Metrics:")
                    print(metrics)
                    metrics.update(best_params)
                    metrics['dataset'] = dataset_name
                    metrics['time'] = train_time

                # Save the metrics
                with open(os.path.join(args['results_folder'], dataset_name, model_name + '.pkl'), 'wb') as f:
                    pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Show results
    results_table, table_col_names = get_results_table()
    mrr_all = get_mrr_models(results_table, table_col_names, args)  # Compute MRR for each metric among all models
    get_mrr_datasets(mrr_all, args)  # Compute MRR for each model among all datasets
