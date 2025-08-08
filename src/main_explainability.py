# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/08/2025

# Package imports
import os
import sys
import sympy
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from kan import ex_round
from copy import deepcopy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from src.data import load_data
from src.utils import get_config
from src.models.models import Kan_model
from src.models.models_utils import get_metrics
from src.representation import plot_binary_explanation_plot, plot_sorted_variances, radar_factory


def run_kan_gam(model, x_train, x_test, y_train, y_test, results_folder, dataset):
    r = model.run_model(x_train, x_test, y_train, y_test)
    pred_proba_test = model.predict_proba(x_test.to_numpy())
    pred_proba_train = model.predict_proba(x_train.to_numpy())
    y_pred_test = model.predict(x_test.to_numpy())
    y_pred_train = model.predict(x_train.to_numpy())
    if pred_proba_test.shape[1] == 2:
        binary = True
        n_classes = 2
        n_logits = 1
    else:
        binary = False
        n_classes = pred_proba_test.shape[1]
        n_logits = n_classes
    metrics_test = get_metrics(y_test.to_numpy(), y_pred_test, pred_proba_test)
    metrics_train = get_metrics(y_train.to_numpy(), y_pred_train, pred_proba_train)

    # Rename y_train to have a "class_target" column
    y_train = pd.DataFrame(y_train.values, columns=['class_target'])
    y_test = pd.DataFrame(y_test.values, columns=['class_target'])

    if binary:
        plot_binary_explanation_plot(y_test,
                                     pred_proba_test[:, 1],
                                     ['0', '1'],
                                     0.5,
                                     os.path.join(results_folder, dataset, 'prob_plot.png'),
                                     title='Probability of positive class')
        plt.close()
    model.model.plot(folder=os.path.join(results_folder, dataset, 'kan'),
                     in_vars=x_train.columns.tolist(),
                     out_vars=[f'logit_{i}' for i in range(n_classes)],
                     varscale=0.2,
                     scale=1)
    plt.savefig(os.path.join(results_folder, dataset, 'kan.png'), bbox_inches='tight', dpi=300)
    plt.close()

    return binary, n_logits, metrics_train, metrics_test, y_train, y_test


def get_patient_values(delta_formula, x_in):
    x = x_in.to_numpy()
    if isinstance(delta_formula, sympy.Float):  # WE have a constant!
        delta = np.zeros((x.shape[0], x.shape[1] + 1))
        delta[:, -1] = float(delta_formula)  # There is only a constant term!
    else:
        delta = np.zeros(
            (x.shape[0], x.shape[1] + 1))  # One input per covariate, one extra output for the constant term
        for i in range(x.shape[0]):  # For each patient
            for fs in delta_formula.args:
                formula_sum_term = deepcopy(fs)
                if isinstance(formula_sum_term, sympy.Float):  # We have a constant!
                    delta[i, -1] = float(formula_sum_term)
                else:  # Since it is a KAAM, it depends on a single variable
                    assert len(formula_sum_term.free_symbols) == 1
                    variable_in_the_expresion = list(formula_sum_term.free_symbols)[0]
                    variable_index = x_in.columns.get_loc(str(variable_in_the_expresion))
                    delta[i, variable_index] += float(
                        formula_sum_term.subs(variable_in_the_expresion, x[i, variable_index]))
    delta = pd.DataFrame(delta, columns=x_in.columns.tolist() + ['const'])
    return delta


def adjust_polynomial(x_train, y_train, x_test, y_test, metrics_train, metrics_test, binary, n_logits, results_folder,
                      dataset):
    # lib = ['x', 'x^2', 'x^3', 'x^4', 'x^5']  # Polynomial model, used because it tends to provide a nice symbolic formula
    model.model(model.dataset['train_input'])  # To have activations updated!
    model.model.auto_symbolic()  # Add the lib in here if desired
    formula = model.model.symbolic_formula()[0]  # We have several formulae, one per logit!
    formula = [ex_round(f, 3) for f in formula]  # Number of digits to approximate

    if binary:
        delta_formula = [formula[1] - formula[0]]  # Keep a single formula (this is the delta)
    else:
        delta_formula = [f for f in
                         formula]  # Keep all the formulae, one per logit. IMPORTANT NOTE: We could simplify the formulae, but this may get rid of the additive separability property!!

    # In the delta_formula, replace x_i by its name
    for i, col in enumerate(x_train.columns):
        for j in range(n_logits):
            delta_formula[j] = delta_formula[j].subs(sympy.symbols(f'x_{i + 1}'), sympy.symbols(col))

    # Save formula as a text file
    with open(os.path.join(results_folder, dataset, 'formula.txt'), 'w') as f:
        for i in range(n_logits):
            f.write(f"Logit {i}: {delta_formula[i]}\n")
            f.write(f"Logit {i} Latex: {sympy.latex(delta_formula[i])}\n")

    # Since the formula may have pruned variables, we keep only the variables that are present in the formula
    actual_vars = []
    for f in delta_formula:
        actual_vars += [str(s) for s in f.free_symbols]
    actual_vars = list(set(actual_vars))  # Remove duplicates
    x_train = x_train[actual_vars]
    x_test = x_test[actual_vars]

    delta_train, delta_test = [], []
    for i in range(n_logits):
        d = get_patient_values(delta_formula[i], x_train)
        delta_train.append(d)
        d.to_csv(os.path.join(results_folder, dataset, f'delta_train_{i}.csv'), index=False)
        d = get_patient_values(delta_formula[i], x_test)
        delta_test.append(d)
        d.to_csv(os.path.join(results_folder, dataset, f'delta_test_{i}.csv'), index=False)
    if binary:
        proba_train_numpy = 1 / (1 + np.exp(-delta_train[0].sum(axis=1).values))
        proba_test_numpy = 1 / (1 + np.exp(-delta_test[0].sum(axis=1).values))
        values_train_numpy = (proba_train_numpy > 0.5).astype(int)
        values_test_numpy = (proba_test_numpy > 0.5).astype(int)
    else:
        proba_train_numpy = np.array(
            [np.exp(np.array(d).sum(axis=1)) / np.sum(np.exp(np.array(delta_train).sum(axis=2)), axis=0) for d
             in delta_train]).T
        proba_test_numpy = np.array(
            [np.exp(np.array(d).sum(axis=1)) / np.sum(np.exp(np.array(delta_test).sum(axis=2)), axis=0) for d in
             delta_test]).T
        values_train_numpy = np.argmax(proba_train_numpy, axis=1)
        values_test_numpy = np.argmax(proba_test_numpy, axis=1)

    metrics_train_numpy = get_metrics(y_train, values_train_numpy, proba_train_numpy)
    metrics_test_numpy = get_metrics(y_test, values_test_numpy, proba_test_numpy)

    print(f"Train metrics without formula: {metrics_train}")
    print(f"Test metrics without formula: {metrics_test}")
    print(f"Train metrics with formula: {metrics_train_numpy}")
    print(f"Test metrics with formula: {metrics_test_numpy}")

    # Save metrics results in df
    metrics_df = pd.DataFrame([metrics_train, metrics_test, metrics_train_numpy, metrics_test_numpy],
                              index=['train', 'test', 'train_formula', 'test_formula'])
    metrics_df.to_csv(os.path.join(args['results_folder'], dataset, 'metrics.csv'))

    if binary:
        plot_binary_explanation_plot(y_test,
                                     proba_test_numpy,
                                     ['0', '1'],
                                     0.5,
                                     os.path.join(args['results_folder'], dataset, 'prob_plot_formula.png'),
                                     title='Probability of positive class')
        plt.close()

    print(f"Variables in the formula: {len(actual_vars)}, which are: {actual_vars}")
    for i in range(n_logits):
        print(f"Formula for logit {i}: {delta_formula[i]}")

    return delta_formula, delta_train, delta_test, proba_train_numpy, proba_test_numpy, x_train, x_test


if __name__ == '__main__':
    # Get the configuration
    args = get_config('explainability')

    for dataset in args['datasets']:
        # Check if the best model is saved
        if os.path.exists(os.path.join(args['base_folder'], 'results_metrics', dataset, 'kan_gam.pkl')):
            with open(os.path.join(args['base_folder'], 'results_metrics', dataset, 'kan_gam.pkl'), 'rb') as f:
                metrics = pickle.load(f)
            print(f"Model for dataset {dataset} found. Using parameters:")
            print(metrics)
            model = Kan_model(hidden_dim=metrics['hidden_dim'],
                              batch_size=metrics['batch_size'],
                              grid=metrics['grid'],
                              k=metrics['k'],
                              seed=metrics['seed'],
                              lr=metrics['lr'],
                              early_stop=metrics['early_stop'],
                              steps=metrics['steps'],
                              lamb=metrics['lamb'],
                              lamb_entropy=metrics['lamb_entropy'],
                              weight=metrics['weight'],
                              sparse_init=metrics['sparse_init'],
                              mult_kan=metrics['mult_kan'])
        else:
            model = Kan_model()
            print(f"Model for dataset {dataset} not found. Using default parameters.")

        # Load the data and run model
        x_train, x_test, y_train, y_test = load_data(dataset, args)
        binary, n_logits, metrics_train, metrics_test, y_train, y_test = run_kan_gam(model,
                                                                                     x_train,
                                                                                     x_test,
                                                                                     y_train,
                                                                                     y_test,
                                                                                     args['results_folder'],
                                                                                     dataset)

        # Adjust a polynomial model
        delta_formula, delta_train, delta_test, proba_train_numpy, proba_test_numpy, x_train, x_test = adjust_polynomial(
            x_train,
            y_train,
            x_test,
            y_test,
            metrics_train,
            metrics_test,
            binary,
            n_logits,
            args[
                'results_folder'],
            dataset)

        # Plot of the sorted variances in training and testing for each logit (feat imp is the variance of the delta vals)
        plot_sorted_variances(x_train, x_test, binary, delta_train, delta_test, n_logits, args, dataset)

        ##### PATIENTS #####
        # Create a folder for the patients
        patients_results_folder = os.path.join(args['results_folder'], dataset, 'patients')
        os.makedirs(patients_results_folder, exist_ok=True)

        if binary:  # Add a dimension to the proba arrays
            proba_train_numpy = proba_train_numpy[:, None]
            proba_test_numpy = proba_test_numpy[:, None]

        n_dists = args['n_dists']
        max_atribs_radar = args['max_atribs_radar']
        max_pats_to_save = args['max_pats_to_save']
        max_plot_curves = args['max_plot_curves']
        for l in range(n_logits):
            logit = 1 if binary else l
            print(f"Processing logit {logit}")

            # Get the most important features for the radar plot, be careful to use only training data!
            variances = delta_train[l].var(axis=0)
            all_cols = delta_train[l].columns.tolist()
            idx_vars = np.argsort(variances.values)[::-1]
            num_of_zero_var = (variances < 1e-6).sum()
            idx_vars = idx_vars[:-num_of_zero_var]

            if delta_train[l].shape[1] > max_atribs_radar:
                idx_vars = idx_vars[:max_atribs_radar]

            for i in tqdm(range(min(x_test.shape[0], max_pats_to_save))):
                actual_label = y_test.iloc[i]['class_target']
                current_patient_info = np.concatenate((x_test.iloc[i].values, [proba_test_numpy[i][l], actual_label]))

                # Find the n_dists closest patients in the training set
                dists = np.linalg.norm(delta_train[l] - delta_test[l].iloc[i], axis=1)
                idx_closest = np.argsort(dists)[:n_dists].tolist()
                pred_prob = (1 / (1 + np.exp(-delta_train[l].iloc[idx_closest].sum(axis=1)))).values
                real_label = y_train.iloc[idx_closest].values
                closest_data = x_train.iloc[idx_closest].values
                closest_data = np.concatenate((closest_data, pred_prob[:, None], real_label), axis=1)
                closest_data = np.vstack(
                    (current_patient_info[None, :], closest_data))  # Add the current patient as the first row
                new_df = pd.DataFrame(closest_data, columns=x_train.columns.tolist() + ['pred_prob', 'real_label'])

                # Limit all new_df values to having 3 decimal numbers at most
                new_df = new_df.map(lambda x: round(x, 3) if isinstance(x, float) else x)
                new_df.to_csv(os.path.join(patients_results_folder, f'patient_{i}_closest_{n_dists}_logit_{logit}.csv'),
                              index=False)

                # Prepare the radar plot, show only the attributes with highest variance
                # Change the order of idx_vars and cols_vars to have the importance in clockwise order in the plot
                idx_vars = idx_vars[::-1]
                cols_vars = [all_cols[i] for i in idx_vars.tolist()]

                n_feats = min(max_atribs_radar, len(cols_vars))  # Number of features to show in the radar plot
                if n_feats >= 3:  # We need at least 3 features to plot a proper radar plot
                    theta = radar_factory(n_feats, frame='polygon')
                    if binary:
                        avg_proba = 1 / (1 + np.exp(-delta_train[l].mean(axis=0).sum())) * np.ones(
                            len(cols_vars))  # Average probability (i.e., "average patient")
                        title = f"Patient {i} results {proba_test_numpy[i][l]:.3f}/{actual_label}, avg proba: {avg_proba[0]:.3f}"
                    else:
                        avg_proba = np.exp(delta_train[l].mean(axis=0).sum()) / sum(
                            [np.exp(d.mean(axis=0).sum()) for d in delta_train]) * np.ones(len(cols_vars))
                        title = f"Patient {i} results {proba_test_numpy[i][l]:.3f}/{actual_label}, avg proba: {avg_proba[0]:.3f}"

                    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
                    # fig.subplots_adjust(top=0.85, bottom=0.05)
                    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
                    ax.set_title(title, position=(0.5, 1.1), ha='center', fontsize=22)

                    # Plot the average of all train patients
                    _ = ax.plot(theta, avg_proba, label='avg', color='b')
                    ax.fill(theta, avg_proba, alpha=0.1, color='b')

                    # Prepare for individual patient plotting
                    avg_delta = delta_train[l].mean(axis=0).values[None, :]
                    avg_matrix = np.repeat(avg_delta, delta_train[l].shape[1], axis=0)

                    # Plot the closest patients
                    for j in range(n_dists):
                        np.fill_diagonal(avg_matrix, delta_train[l].iloc[idx_closest[j]].values)
                        if binary:
                            pat_proba = 1 / (1 + np.exp(-avg_matrix.sum(axis=1)))
                        else:
                            den_term = np.zeros(delta_train[0].shape[1])
                            for ll in range(n_logits):
                                den_matrix = np.repeat(delta_train[ll].mean(axis=0).values[None, :],
                                                       delta_train[ll].shape[1],
                                                       axis=0)
                                np.fill_diagonal(den_matrix, delta_train[ll].iloc[idx_closest[j]].values)
                                den_term += np.exp(den_matrix.sum(axis=1))
                            pat_proba = np.exp(avg_matrix.sum(axis=1)) / den_term
                        _ = ax.plot(theta, pat_proba[idx_vars], label=f'closest_{j}', color='g', alpha=0.5)
                        ax.fill(theta, pat_proba[idx_vars], alpha=0.1, color='g')

                    # Plot the current patient last, so that it can be seen better
                    np.fill_diagonal(avg_matrix, delta_test[l].iloc[i].values)
                    if binary:
                        pat_proba = 1 / (1 + np.exp(-avg_matrix.sum(axis=1)))
                    else:
                        den_term = np.zeros(delta_train[0].shape[1])
                        for ll in range(n_logits):
                            den_matrix = np.repeat(delta_train[ll].mean(axis=0).values[None, :],
                                                   delta_train[ll].shape[1],
                                                   axis=0)
                            np.fill_diagonal(den_matrix, delta_test[ll].iloc[i].values)
                            den_term += np.exp(den_matrix.sum(axis=1))
                        pat_proba = np.exp(avg_matrix.sum(axis=1)) / den_term
                    _ = ax.plot(theta, pat_proba[idx_vars], label='test_patient', color='r')
                    ax.fill(theta, pat_proba[idx_vars], alpha=0.1, color='r')

                    ax.set_varlabels(cols_vars, fontsize=18)
                    plt.rcParams.update({
                        'axes.titlesize': 22,  # Title size of axes
                        'axes.labelsize': 20,  # x and y axis labels
                        'xtick.labelsize': 18,  # x tick labels
                        'ytick.labelsize': 18,  # y tick labels
                        'legend.fontsize': 18,  # Legend font
                    })
                    # plt.legend(loc='best')  # Note: this can be uncommented, but may clutter the plot
                    plt.savefig(os.path.join(patients_results_folder, f'patient_{i}_radar_logit{logit}.png'),
                                bbox_inches='tight',
                                dpi=600)
                    plt.close()

                # Curves plot: show only the ones that do matter!!
                # Revert again the order to have the right plot order
                idx_vars = idx_vars[::-1]
                cols_vars = [all_cols[i] for i in idx_vars.tolist()]

                n_feats = min(len(cols_vars), max_plot_curves)  # Number of features to show in the curves plot
                if n_feats > 0:  # There is something to show
                    if binary:
                        fig, axs = plt.subplots(n_feats + 1, 1, figsize=(12, 3 * (n_feats + 1)))
                        x_vals = np.arange(delta_train[l].sum(axis=1).min(), delta_train[l].sum(axis=1).max(), 0.01)
                        theor_proba = 1 / (1 + np.exp(-x_vals))
                        axs[0].plot(x_vals, theor_proba, 'b', alpha=0.2)
                        axs[0].scatter(delta_test[l].sum(axis=1)[i], proba_test_numpy[i][l], color='r')
                        axs[0].set_xlabel('Logit', fontsize=20)
                        axs[0].set_ylabel('Probability', fontsize=20)
                        axs[0].set_title(f'Patient {i}', fontsize=22)
                    else:
                        fig, axs = plt.subplots(n_feats, 1, figsize=(12, 3 * n_feats))

                    for idj, feat_name in enumerate(cols_vars):
                        if idj < n_feats:  # Only plot the first n_feats features
                            if binary:
                                j = idj + 1  # The first plot is already used for the theoretical curve
                            else:
                                j = idj
                            # Keep only unique values of x_test[feat_name]
                            idxs = np.unique(x_train[feat_name].values, return_index=True)[1]
                            if n_feats > 1:  # Multiple plots
                                axs[j].plot(x_train[feat_name].values[idxs],
                                            delta_train[l][feat_name].values[idxs],
                                            color='b')
                                axs[j].scatter(x_train[feat_name].values[idxs],
                                               delta_train[l][feat_name].values[idxs],
                                               color='b',
                                               alpha=0.2)
                                for jj in range(n_dists):
                                    axs[j].scatter(x_train.iloc[idx_closest[jj]][feat_name],
                                                   delta_train[l].iloc[idx_closest[jj]][feat_name],
                                                   color='g',
                                                   alpha=1,
                                                   s=90)

                                axs[j].scatter(x_test[feat_name].values[i],
                                               delta_test[l][feat_name].values[i],
                                               color='r', s=30)
                                axs[j].set_ylabel(f"{feat_name}", fontsize=20)
                            else:  # Single plot
                                axs.plot(x_train[feat_name].values[idxs],
                                         delta_train[l][feat_name].values[idxs],
                                         color='b')
                                axs.scatter(x_train[feat_name].values[idxs],
                                            delta_train[l][feat_name].values[idxs],
                                            color='b',
                                            alpha=0.2)
                                for jj in range(n_dists):
                                    axs.scatter(x_train.iloc[idx_closest[jj]][feat_name],
                                                delta_train[l].iloc[idx_closest[jj]][feat_name],
                                                color='g',
                                                alpha=1,
                                                s=90)

                                axs.scatter(x_test[feat_name].values[i],
                                            delta_test[l][feat_name].values[i],
                                            color='r', s=30)
                                axs.set_ylabel(f"{feat_name}", fontsize=20)

                    plt.rcParams.update({
                        'axes.titlesize': 22,  # Title size of axes
                        'axes.labelsize': 20,  # x and y axis labels
                        'xtick.labelsize': 18,  # x tick labels
                        'ytick.labelsize': 18,  # y tick labels
                        'legend.fontsize': 18,  # Legend font
                    })
                    plt.savefig(os.path.join(patients_results_folder, f'patient_{i}_curves_logit_{logit}.png'),
                                bbox_inches='tight',
                                dpi=600)
                    plt.close()
