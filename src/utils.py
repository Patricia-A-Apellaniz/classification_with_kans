# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 31/07/2025


# Package imports
import os


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
    datasets = 'all'
    if datasets == 'all':
        args['datasets'] = ['heart', 'diabetes_h', 'diabetes_130', 'obesity', 'obesity_bin', 'breast_cancer']
    else:
        args['datasets'] = [datasets]

    # Set the path to the results folder
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    args['base_folder'] = base_folder
    args['data_folder'] = os.path.join(base_folder, 'data')
    results_folder = 'results_metrics' if task == 'metrics' else 'results_explainability_bigger_size'
    args['results_folder'] = os.path.join(base_folder, results_folder)
    create_results_folder(args['results_folder'], args)

    # Training parameters
    args['train'] = not True  # If True, train the models. If False, load the models from disk
    args['n_folds'] = 3  # Number of folds for cross-validation
    args['n_jobs'] = 1  # Number of jobs to run in parallel. Unsure whether the code is ready to parallelize the KAN and NAM in its current form...

    # Explainability parameters (representation info)
    args['n_dists'] = 5  # Representation parameter: threshold on the number of closest patients to show
    args['max_atribs_radar'] = 10  # Max # of attributes to show in the radar plot, filtered by variance (in case there are more covariates than this number)
    args['max_plot_curves'] = 5  # Max # of curves to plot in the partial dependence plots
    args['max_pats_to_save'] = 10  # Max # of patients to save in the patients folder

    return args
