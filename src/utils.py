# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 31/07/2025


# Package imports
import os
import sys
import pickle

import numpy as np
import scipy.stats as stats
import statsmodels.stats.multitest as multitest

from tabulate import tabulate

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from src.models.model_utils import get_model


#####################################
### ENVIRONMENT AND CONFIGURATION ###
#####################################
def create_results_folder(results_folder, args):
    for dataset in args['datasets']:
        folder_path = os.path.join(results_folder, dataset)
        os.makedirs(folder_path, exist_ok=True)


def get_config(task):
    args = {}

    # Model selection
    # Options are 'all', 'mlp', 'lr', 'rf', 'nam', 'kan', 'kan_gam'
    models = 'all'
    if models == 'all':
        args['models'] = ['mlp', 'lr', 'rf', 'nam', 'kan', 'kan_gam']
    else:
        args['models'] = [models]

    # Dataset selection
    # Options are 'all', 'heart', 'diabetes_h', 'diabetes_130', 'obesity', 'obesity_bin', 'breast_cancer'
    datasets = 'diabetes_130'
    if datasets == 'all':
        args['datasets'] = ['heart', 'diabetes_h', 'diabetes_130', 'obesity', 'obesity_bin', 'breast_cancer']
    else:
        args['datasets'] = [datasets]

    # Set the path to the results folder
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    args['base_folder'] = base_folder
    args['data_folder'] = os.path.join(base_folder, 'data')
    results_folder = 'results_performance' if task == 'performance' else 'results_interpretability'
    args['results_folder'] = os.path.join(base_folder, results_folder)

    # Training parameters
    args['train'] = True  # If True, train the models. If False, load the models from disk
    args['n_folds'] = 3  # Number of folds for cross-validation
    args['n_jobs'] = 1  # Number of jobs to run in parallel

    # Interpretability parameters (representation info)
    args['n_dists'] = 5 # Representation parameter: threshold on the number of closest patients to show
    args['max_atribs_radar'] = 10  # Max # of attributes to show in the radar plot, filtered by variance
    args['max_plot_curves'] = 5  # Max # of curves to plot in the partial dependence plots
    args['max_pats_to_save'] = 5  # Max # of patients to save in the patients folder

    return args


#####################################
###    RESULTS REPRESENTATION     ###
#####################################

def get_results_table(args):
    table_col_names = ['Dataset', 'Model', 'Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Time']
    results_table = []
    results_table_no_ci = []
    for dataset in args['datasets']:
        models = args['models'].copy()
        if dataset not in ['heart', 'obesity_bin', 'breast_cancer']:
            models.remove('nam')
        for model_name in models:
            # Load the metrics
            with open(os.path.join(args['results_folder'], dataset, model_name + '.pkl'), 'rb') as f:
                metrics = pickle.load(f)

            # Approximate all metrics to two decimal places
            no_ci_metrics = {'accuracy': 0.0, 'roc_auc': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
            for key in ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']:
                # metrics[key] = np.round(metrics[key], 2)
                no_ci_metrics[key] = metrics[key]['mean']
                metrics[key] = str(np.round(metrics[key]['mean'], 2)) + " (" + str(
                    np.round(metrics[key]['CI_95%'][0], 3)) + ", " + str(np.round(metrics[key]['CI_95%'][1], 3)) + ")"

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

            results_table_no_ci.append([dataset,
                                        model_name,
                                        no_ci_metrics['accuracy'],
                                        no_ci_metrics['roc_auc'],
                                        no_ci_metrics['f1'],
                                        no_ci_metrics['precision'],
                                        no_ci_metrics['recall'],
                                        avg_time])
    print('\n\n---------------Performance results---------------\n')
    # print(tabulate(results_table, headers=table_col_names, tablefmt='latex', floatfmt=".2f"))
    print(tabulate(results_table, headers=table_col_names, floatfmt=".2f"))

    return results_table, table_col_names, results_table_no_ci


def friedman_test(all_data, comp_index, alpha, higher_is_better):
    """
    Perform the Friedman test on the provided data. Based on Demsar06.
    :param all_data: 2D numpy array of shape (n_methods, n_datasets) where each row is a method and each column is a dataset.
    :param comp_index: Method to set as baseline for post-hoc tests, as in Demsar06. Should be the best performing metric...
    :param alpha: significance level for the test.
    :return: Friedman test p_value, davenport p-value and pairwise to the best baseline post-hoc p-values.
    """
    # Check that comp_index gives the best performing method (double check just in case...)
    avg_performance = np.mean(all_data, axis=1)  # Average performance across datasets for each method
    if higher_is_better:
        assert comp_index == np.argmax(avg_performance), "comp_index must be the index of the best performing method."
    else:
        assert comp_index == np.argmin(avg_performance), "comp_index must be the index of the best performing method."
    # Manual implementation of the Friedman test--to compute post-hoc metrics later on
    n_methods, n_reps = all_data.shape
    ranking_matrix = np.zeros_like(all_data)

    for k in range(n_reps):
        # Rank the methods for each dataset/fold
        if higher_is_better:
            ranking_matrix[:, k] = stats.rankdata(-all_data[:, k], method='average')  # Average ranks for ties
        else:
            ranking_matrix[:, k] = stats.rankdata(all_data[:, k], method='average')  # Average ranks for ties

    # Calculate the Friedman test statistic
    average_rank = np.mean(ranking_matrix, axis=1)
    friedman_stat = (12 * n_reps / (n_methods * (n_methods + 1))) * (
                np.sum(np.square(average_rank)) - (n_methods * (n_methods + 1) ** 2 / 4))  # Friedman test statistic
    friedman_p_value = stats.chi2.sf(friedman_stat, df=n_methods - 1)  # p-value for the Friedman test
    davenport_stat = friedman_stat * (n_reps - 1) / (n_reps * (n_methods - 1))  # Davenport's statistic
    davenport_p_value = stats.f.sf(davenport_stat, dfn=n_methods - 1, dfd=(n_methods - 1) * (n_methods) * (n_reps - 1))

    # If we reject, we can perform post-hoc tests here. # TODO: Unsure if this is OK, need to account for higher is better in the p-values!!
    z_stat = np.zeros(n_methods)
    for j in range(n_methods):
        z_stat[j] = (average_rank[comp_index] - average_rank[j]) / np.sqrt(
            (n_methods * (n_methods + 1)) / (6 * n_reps))  # Z-statistic for post-hoc tests
    p_values_post_hoc = stats.norm.cdf(z_stat)
    _, p_values_adjusted_post_hoc, _, _ = multitest.multipletests(p_values_post_hoc, alpha=alpha,
                                                                  method='holm')  # Holm-Bonferroni correction
    return friedman_p_value, davenport_p_value, p_values_post_hoc, p_values_adjusted_post_hoc


def get_p_values_from_table_data(data, alpha=0.05, higher_is_better=True, output_latex=False, list_of_methods=None,
                                 list_of_metrics=None):
    """
    Function to get p-values from a table of data in a structured way, automatically comparing with the best method for each metric.
    :param data: Organized as a numpy array: methods_to_compare x metrics x datasets/folds. Note that all datasets/folds need to have the same ordering: we use paired tests!!
    :param alpha: float, significance level for the hypothesis test.
    :param higher_is_better: bool or list of bool, if True, higher values are better, otherwise lower values are better.
    :param output_latex: bool, if True, outputs the table in LaTeX format, to copy and paste into a LaTeX document.
    :param list_of_methods: List of method names, if None, uses the default names.
    :param list_of_metrics: List of metric names, if None, uses the default names.
    :return: Outputs a p-value table comparing each method to the specified comparison method.
    """

    assert isinstance(data, np.ndarray), "Data must be a numpy array."
    assert data.ndim == 3, "Data must be a 3D numpy array with shape (n_methods, n_metrics, n_reps)."

    n_methods, n_metrics, n_reps = data.shape
    average_results = np.nanmean(data,
                                 axis=2)  # Average over repetitions, we have an array of shape (n_methods, n_metrics)

    if list_of_methods is None:
        list_of_methods = [f'Method {i + 1}' for i in range(data.shape[0])]
    if list_of_metrics is None:
        list_of_metrics = [f'Metric {i + 1}' for i in range(data.shape[1])]

    if not isinstance(higher_is_better, bool):
        assert len(
            higher_is_better) == n_metrics, "If higher_is_better is a list, it must have the same length as the number of metrics."
    else:
        higher_is_better = [higher_is_better] * data.shape[1]  # If it's a single bool, replicate it for all metrics

    max_idxs = np.argmax(average_results, axis=0)
    min_idxs = np.argmin(average_results, axis=0)
    comp_index = [max_idxs[i] if higher_is_better[i] else min_idxs[i] for i in range(n_metrics)]

    for i in range(n_metrics):

        # Print the data for complete reference
        print(f'\n---------------Data for metric {list_of_metrics[i]}, where higher_is_better is {higher_is_better[i]}---------------')
        for j in range(n_methods):
            vals_to_show = data[j, i, :]
            vals_to_show_str = ', '.join([f"{v:.3f}" if not np.isnan(v) else "nan" for v in vals_to_show])
            print(f'{list_of_methods[j]}: [{vals_to_show_str}] / avg: {np.nanmean(data[j, i, :]):.3f}')
        table_metrics = ['Average metric'] + [f"{np.nanmean(data[j, i, :]):.3f}" for j in range(n_methods)]
        # First method: use paired Wilcoxon signed-rank test to obtain p-values, and correct them using Holm-Bonferroni method. This is done per-metric, so if we have many metrics, we will have many p-values.
        baseline_values = data[comp_index[i], i, :]  # Baseline values for the metric
        # remove nan values from baseline_values
        baseline_values = baseline_values[~np.isnan(baseline_values)]
        p_values = []
        for j in range(n_methods):
            test_values = data[j, i, :]  # Test values for the metric
            # remove nan values from test_values
            test_values = test_values[~np.isnan(test_values)]
            # If the lengths of baseline_values and test_values are different, we need to remove the corresponding values from both
            # min_length = min(len(baseline_values), len(test_values))

            if comp_index[
                i] == j:  # If we are comparing the baseline method with itself, we skip this comparison, as the Wilcoxon test will throw an error
                p_values.append(1.0)  # No difference, p-value is 1
                continue
            if higher_is_better[i]:
                # If higher is better, we want to test if the test values are significantly lower than the baseline values (i.e., significantly worse)
                _, p_value = stats.wilcoxon(test_values, baseline_values, alternative='less')
            else:
                # If lower is better, we want to test if the test values are significantly higher than the baseline values (i.e., significantly worse)
                _, p_value = stats.wilcoxon(test_values, baseline_values, alternative='greater')
            p_values.append(p_value)
        # Apply Holm-Bonferroni correction
        print('\n')
        p_values = np.array(p_values)
        _, corrected_p_vals, _, _ = multitest.multipletests(np.array(p_values), alpha=alpha, method='holm')
        # Prepare a table to store all data for this metric
        table_wilcoxon_corr = ['Paired Wilcoxon tests (corrected)']
        table_wilcoxon_unc = ['Paired Wilcoxon tests (uncorrected)']
        for j in range(n_methods):
            p_val_str = f"{corrected_p_vals[j]:.3f}" if corrected_p_vals[j] >= 1e-3 else "<1e-3"  # Format p-values
            if corrected_p_vals[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            table_wilcoxon_corr.append(p_val_str)

            p_val_str = f"{p_values[j]:.3f}" if p_values[j] >= 1e-3 else "<1e-3"  # Format small p-values
            if p_values[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            table_wilcoxon_unc.append(p_val_str)

        # Second method: the Friedman test, which is a non-parametric test for repeated measures done on all metrics at once. Blocks = methods, treatments = datasets / folds (we could also implement one on datasets * metrics, a general one, later on). We rely on Demsar06 for this implementation.
        fr_data = data[:, i, :]
        # remove columns with nan values
        fr_data = fr_data[:, ~np.isnan(fr_data).any(axis=0)]
        friedman_p_value, davenport_p_value, p_values_post_hoc_unc, p_values_post_hoc_corr = friedman_test(fr_data, comp_index[i], alpha, higher_is_better[i])

        # Prepare this for the table
        friedman_post_hoc_table_corr = ['Friedman post-hoc tests (Corrected)']
        friedman_post_hoc_table_unc = ['Friedman post-hoc tests (Uncorrected)']
        for j in range(n_methods):
            p_val_str = f"{p_values_post_hoc_corr[j]:.3f}" if p_values_post_hoc_corr[
                                                                  j] >= 1e-3 else "<1e-3"  # Format p-values
            if p_values_post_hoc_corr[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            friedman_post_hoc_table_corr.append(p_val_str)

            p_val_str = f"{p_values_post_hoc_unc[j]:.3f}" if p_values_post_hoc_unc[
                                                                 j] >= 1e-3 else "<1e-3"  # Format small p-values
            if p_values_post_hoc_unc[j] >= alpha:
                p_val_str += '*'  # Mark best values
            if j == comp_index[i]:
                p_val_str += ' (baseline)'  # Mark the baseline method
            friedman_post_hoc_table_unc.append(p_val_str)
        if friedman_p_value < 1e-3:
            friedman_p_value_str = "<1e-3"  # Format small p-values
        else:
            friedman_p_value_str = f"{friedman_p_value:.3f}"
        if davenport_p_value < 1e-3:
            davenport_p_value_str = "<1e-3"  # Format small p-values
        else:
            davenport_p_value_str = f"{davenport_p_value:.3f}"
        print(
            f'Friedman p-value: {friedman_p_value_str}, Davenport p-value: {davenport_p_value_str} for metric {list_of_metrics[i]}')
        if n_reps <= 10 or n_methods <= 5:
            print(
                'Since the number of data points is small, the Friedman test may not be reliable. Consider using a larger dataset or a different test.')

        table_data = [table_metrics, table_wilcoxon_unc, table_wilcoxon_corr, friedman_post_hoc_table_unc,
                      friedman_post_hoc_table_corr]

        if output_latex:
            print(tabulate(table_data, headers=[f'Metric {list_of_metrics[i]}'] + list_of_methods, tablefmt='latex'))
        else:
            print(tabulate(table_data, headers=[f'Metric {list_of_metrics[i]}'] + list_of_methods, tablefmt='grid'))

    # Finally, run a Friedman test on all metrics at once
    all_data = data.copy()
    # For all metrics where lower is better, we need to invert the data so that higher is better always
    for j in range(n_metrics):
        if not higher_is_better[j]:
            all_data[:, j, :] = -all_data[:, j, :]  # Invert the data for lower is better metrics
    # Now, reshape the data to have shape (n_methods, n_metrics * n_reps)
    all_data = all_data.reshape(n_methods, n_metrics * n_reps)
    avg_metrics = np.nanmean(all_data, axis=1)  # Average over repetitions and metrics
    best_method = np.argmax(avg_metrics)  # Best method across all metrics (remember, higher is better now!)
    friedman_p_value, davenport_p_value, p_values_post_hoc_unc, p_values_post_hoc_corr = friedman_test(all_data,
                                                                                                       best_method,
                                                                                                       alpha,
                                                                                                       higher_is_better=True)
    print(f'\n\n-------Friedman test on all metrics-------\np-value: {friedman_p_value:.4f}, Davenport p-value: {davenport_p_value:.4f}')
    # Prepare this for the table
    friedman_post_hoc_table_unc = ['Friedman post-hoc tests (all metrics, uncorrected)']
    friedman_post_hoc_table_corr = ['Friedman post-hoc tests (all metrics, corrected)']
    for j in range(n_methods):
        p_val_str = f"{p_values_post_hoc_corr[j]:.3f}" if p_values_post_hoc_corr[
                                                              j] >= 1e-3 else "<1e-3"  # Format p-values
        if p_values_post_hoc_corr[j] >= alpha:
            p_val_str += '*'  # Mark best values
        if j == best_method:
            p_val_str += ' (baseline)'  # Mark the baseline method
        friedman_post_hoc_table_corr.append(p_val_str)

        p_val_str = f"{p_values_post_hoc_unc[j]:.3f}" if p_values_post_hoc_unc[
                                                             j] >= 1e-3 else "<1e-3"  # Format small p-values
        if p_values_post_hoc_unc[j] >= alpha:
            p_val_str += '*'  # Mark best values
        if j == best_method:
            p_val_str += ' (baseline)'  # Mark the baseline method
        friedman_post_hoc_table_unc.append(p_val_str)
    table_data = [friedman_post_hoc_table_unc, friedman_post_hoc_table_corr]
    if output_latex:
        print(tabulate(table_data, headers=['All metrics'] + list_of_methods, tablefmt='latex'))
    else:
        print(tabulate(table_data, headers=['All metrics'] + list_of_methods, tablefmt='grid'))



#####################################
###        INTERPRETABILITY       ###
#####################################

