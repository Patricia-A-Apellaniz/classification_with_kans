# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 31/07/2025

# Package imports
import argparse
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tueplots import bundles
from tabulate import tabulate

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from src.data import (
    load_data,
    make_sample_splits,
    preprocess_split,
)
from src.models.model_utils import (
    add_bootstrap_metrics,
    fit_with_saved_params,
    get_best_params,
)
from src.utils import get_config, get_results_table, get_p_values_from_table_data, create_results_folder


def get_mrr_datasets(mrr_all_elements):
    for model in args['models']:
        if len(mrr_all_elements[model]) > 0:
            mrr_all_elements[model] = np.round(
                sum([1 / v for v in mrr_all_elements[model]]) / len(mrr_all_elements[model]), 2)
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
            values = [{res[1]: float(res[idx + 2].split(' ')[0])} for res in results_table if
                      res[0] == dataset and float(res[idx + 2].split(' ')[
                                                      0]) > -0.5]  # Note htat -0.5 is just a threshold: we put -1 to flag the metrics that were not computed
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


def get_box_plots(results_folder):
    ranking = {}
    for dataset in args['datasets']:
        if dataset not in ranking:
            ranking[dataset] = {}
        # Go through each dataset and get the ranking of the models for all metrics
        dataset_results = [res for res in results_table if res[0] == dataset]
        for metric_idx in range(2, len(table_col_names) - 1):
            metric_values = []
            for res in dataset_results:
                value = float(res[metric_idx].split(' ')[0])
                if value > -0.5:  # We only consider valid values
                    metric_values.append((res[1], value))  # (model_name, value)
            # Sort by value in descending order
            metric_values = sorted(metric_values, key=lambda x: x[1], reverse=True)
            rank = 1
            for j in range(len(metric_values)):
                if j > 0:
                    if abs(metric_values[j][1] - metric_values[j - 1][1]) > 0.001:
                        rank += 1
                model_name = metric_values[j][0]
                if model_name not in ranking[dataset]:
                    ranking[dataset][model_name] = []
                ranking[dataset][model_name].append(rank)

    # Create boxplots for each model with seaborn
    data = []
    for dataset in args['datasets']:
        if dataset in ranking:
            for model in args['models']:
                if model in ranking[dataset]:
                    for r in ranking[dataset][model]:
                        data.append({'Dataset': dataset, 'Model': model, 'Rank': r})
    df = pd.DataFrame(data)
    df['Dataset'] = df['Dataset'].map({'heart': 'Heart',
                                       'diabetes_h': 'Diabetes-H',
                                       'diabetes_130': 'Diabetes-130',
                                       'obesity': 'Obesity',
                                       'obesity_bin': 'Obesity-Bin',
                                       'breast_cancer': 'Breast-Cancer'})
    df['Model'] = df['Model'].map({'mlp': 'MLP',
                                   'lr': 'LR',
                                   'rf': 'RF',
                                   'nam': 'NAM',
                                   'kan': 'Logistic-KAN',
                                   'kan_gam': 'KAAM'})
    with plt.rc_context({**bundles.icml2024(column='half', nrows=1, ncols=1, usetex=True)}):
        plt.figure(figsize=(6.5, 2))
        palette = sns.color_palette("tab10")
        model_palette = dict(zip(df['Model'].unique(), palette))

        ax = sns.boxplot(
            data=df,
            x='Dataset',
            y='Rank',
            hue='Model',
            palette=model_palette,
            medianprops=dict(color='red', linewidth=2),
            whis=[0, 100],
            fliersize=0
        )
        ax.set_xlabel("")
        for patch, median_line in zip(ax.patches, ax.lines[4::6]):
            facecolor = patch.get_facecolor()
            median_line.set_color(facecolor)
            median_line.set_linewidth(2.5)

        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylabel('Rank')
        ax.grid(axis='y')
        ax.legend(loc='center', bbox_to_anchor=(1.1, 0.5))
        plt.tight_layout()
        plt.savefig(results_folder + os.sep + 'ranking_boxplots.pdf', dpi=300)
        plt.show()
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run one small logistic regression experiment per dataset.',
    )
    cli_args = parser.parse_args()
    args = get_config('performance')

    if cli_args.smoke_test:
        args['train'] = True
        args['hp_search'] = False
        args['models'] = ['lr']
        args['sample_sizes'] = [100]
        args['results_folder'] = os.path.join(
            args['base_folder'], 'results_metrics', 'smoke_test'
        )

    if args['train']:
        for dataset_name in args['datasets']:
            x, y = load_data(
                dataset_name,
                args,
                raw=True,
            )

            test_idx, train_splits = make_sample_splits(
                dataset_name,
                x,
                y,
                args,
            )

            if not train_splits:
                print(
                    f'No selected sizes are available for {dataset_name}.'
                )
                continue

            for train_size, train_idx in sorted(train_splits.items()):
                x_train, x_test, preprocessing = preprocess_split(
                    x.iloc[train_idx],
                    x.iloc[test_idx],
                )

                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                result_folder = os.path.join(
                    args['results_folder'],
                    str(train_size),
                    dataset_name,
                )
                os.makedirs(result_folder, exist_ok=True)

                for model_name in args['models']:
                    if (
                        model_name == 'nam'
                        and len(np.unique(y_train)) > 2
                    ):
                        print(
                            f'Skipping multiclass NAM: '
                            f'{dataset_name}, N={train_size}.'
                        )
                        continue

                    result_path = os.path.join(
                        result_folder,
                        model_name + '.pkl',
                    )

                    if os.path.exists(result_path):
                        print(f'Result already exists: {result_path}')
                        continue

                    print(
                        f'\nTraining {model_name} on '
                        f'{dataset_name}, N={train_size}.'
                    )

                    if args['hp_search']:
                        (
                            best_params,
                            best_model,
                            hp_search_time,
                            training_time,
                        ) = get_best_params(
                            model_name,
                            x_train,
                            y_train,
                            args,
                        )
                    else:
                        (
                            best_params,
                            best_model,
                            hp_search_time,
                            training_time,
                        ) = fit_with_saved_params(
                            model_name,
                            dataset_name,
                            x_train,
                            y_train,
                            args,
                            train_size,
                        )

                    if best_model is None:
                        print(
                            f'No model returned for '
                            f'{dataset_name}/{model_name}.'
                        )
                        continue

                    start = time.perf_counter()

                    y_pred = np.asarray(
                        best_model.predict(x_test)
                    ).reshape(-1)

                    y_proba = np.asarray(
                        best_model.predict_proba(x_test)
                    )

                    inference_time = time.perf_counter() - start

                    classes = np.asarray(
                        best_model.classes_
                    ).reshape(-1)

                    if y_proba.ndim != 2:
                        raise ValueError(
                            f'{dataset_name}/{model_name}: '
                            f'invalid probability shape {y_proba.shape}.'
                        )

                    if y_proba.shape != (len(y_test), len(classes)):
                        raise ValueError(
                            f'{dataset_name}/{model_name}: '
                            'probabilities do not match test rows/classes.'
                        )

                    if len(y_pred) != len(y_test):
                        raise ValueError(
                            f'{dataset_name}/{model_name}: '
                            'predictions do not match test rows.'
                        )

                    result = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'train_size': train_size,
                        'train_indices': train_idx,
                        'test_indices': test_idx,
                        'y_test': np.asarray(y_test).reshape(-1),
                        'y_pred': y_pred,
                        'y_proba': y_proba,
                        'classes': classes,
                        'hyperparameters': best_params,
                        'preprocessing': preprocessing,
                        'split_seed': args['split_seed'],
                        'hp_search': args['hp_search'],
                        'hp_search_time': hp_search_time,
                        'training_time': training_time,
                        'inference_time': inference_time,
                    }

                    if args['calculate_bootstrap']:
                        add_bootstrap_metrics(
                            result,
                            n_bootstrap=args['n_bootstrap'],
                            ci=args['bootstrap_ci'],
                        )

                    with open(result_path, 'xb') as f:
                        pickle.dump(
                            result,
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )

                    print(f'Saved {result_path}')

    # Read one table per size, including when train=False.
    available_sizes = sorted({
        size
        for dataset in args['datasets']
        for size in args['dataset_sizes'][dataset]
        if (
            args['sample_sizes'] is None
            or size in args['sample_sizes']
        )
    })

    for train_size in available_sizes:
        table_args = args.copy()
        table_args['datasets'] = [
            dataset
            for dataset in args['datasets']
            if train_size in args['dataset_sizes'][dataset]
        ]
        table_args['results_folder'] = os.path.join(
            args['results_folder'],
            str(train_size),
        )

        print(f'\nResults for training size N={train_size}')
        get_results_table(table_args)
