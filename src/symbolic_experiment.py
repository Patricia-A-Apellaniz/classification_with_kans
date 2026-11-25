"""Compare spline, symbolic, and two-decimal KAAM predictions on fixed test sets."""

import argparse
import os
import pickle
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import sympy as sp
import torch
from joblib import Parallel, delayed, parallel_config
from kan import ex_round
from scipy.special import softmax
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_data, make_sample_splits, preprocess_split
from src.models.model_utils import fit_with_saved_params, get_metrics, load_configured_params
from src.utils import get_config

VARIANTS = ('original', 'formula', 'formula_2dec')
REFERENCE_SIZES = {'heart': 1000, 'diabetes_h': 1000, 'diabetes_130': 1000,
                   'obesity': 1000, 'obesity_bin': 1000, 'breast_cancer': 455}
OUTPUT_DIR = ROOT / 'results_symbolic'


def log(dataset, message):
    print(f'[{dataset}] {message}', flush=True)


def evaluate_formulas(formulas, symbols, x):
    columns = []
    data = np.asarray(x, dtype=float)
    for formula in formulas:
        fun = sp.lambdify(symbols, formula, modules='numpy')
        with np.errstate(all='ignore'):
            values = np.asarray(fun(*data.T))
        if np.iscomplexobj(values):
            raise ValueError('A symbolic expression produced complex output')
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            values = np.full(len(data), values.item())
        else:
            values = np.broadcast_to(values.reshape(-1), (len(data),))
        columns.append(values)
    logits = np.column_stack(columns)
    if not np.isfinite(logits).all():
        raise ValueError('Symbolic expressions produced non-finite logits on test')
    return logits


def paired_bootstrap(y, classes, predictions, n_bootstrap, seed, n_bins):
    rng = np.random.default_rng(seed)
    distributions = {name: {} for name in VARIANTS}
    for _ in range(n_bootstrap):
        indices = rng.choice(len(y), len(y), replace=True)
        if not np.array_equal(np.unique(y[indices]), classes):
            continue
        for name in VARIANTS:
            pred, proba = predictions[name]
            values = get_metrics(y[indices], pred[indices], proba[indices],
                                 classes=classes, n_bins=n_bins)
            for key, value in values.items():
                distributions[name].setdefault(key, []).append(value)
    n_valid = len(next(iter(distributions['original'].values()), []))
    if not n_valid:
        raise ValueError('No bootstrap replicate contained all classes')
    summaries = {}
    for name, metrics in distributions.items():
        summaries[name] = {
            'metric_mean': {key: float(np.mean(values)) for key, values in metrics.items()},
            'metric_ci': {key: tuple(map(float, np.percentile(values, [2.5, 97.5])))
                          for key, values in metrics.items()},
            'bootstrap_values': {key: np.asarray(values) for key, values in metrics.items()},
        }
    differences = {}
    for name in ('formula', 'formula_2dec'):
        differences[name] = {}
        for key, values in distributions[name].items():
            delta = np.asarray(values) - np.asarray(distributions['original'][key])
            differences[name][key] = {'mean': float(np.mean(delta)),
                                      'ci': tuple(map(float, np.percentile(delta, [2.5, 97.5])))}
    return summaries, differences, n_valid


def check_reference(dataset, size, train_idx, test_idx, y_test, args):
    path = Path(args['results_folder']) / str(size) / dataset / 'kan_gam.pkl'
    if not path.exists():
        log(dataset, 'No main KAAM result found; split cannot be cross-checked.')
        return False
    with path.open('rb') as file:
        result = pickle.load(file)
    expected = {'dataset': dataset, 'model': 'kan_gam', 'train_size': size}
    if not isinstance(result, dict) or any(result.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Reference result metadata does not match: {path}')
    for key, value in (('train_indices', train_idx), ('test_indices', test_idx), ('y_test', y_test)):
        if not np.array_equal(np.asarray(result[key]).reshape(-1), np.asarray(value).reshape(-1)):
            raise ValueError(f'Reference result {key} does not match: {path}')
    return True


def valid_existing(path, dataset, size, train_idx, test_idx, y_test, params, options, args):
    try:
        with path.open('rb') as file:
            saved = pickle.load(file)
        expected = {'dataset': dataset, 'train_size': size, 'round_decimals': 2,
                    'hyperparameters': params, 'split_seed': args['split_seed']}
        if any(saved.get(k) != v for k, v in expected.items()):
            return False
        bootstrap = saved['bootstrap']
        if (bootstrap['n_bootstrap'], bootstrap['seed'], bootstrap['ece_bins']) != (options.n_bootstrap, options.seed, args['ece_bins']):
            return False
        if (saved.get('formula_atol'), saved.get('formula_rtol')) != (options.formula_atol, options.formula_rtol):
            return False
        for key, value in (('train_indices', train_idx), ('test_indices', test_idx), ('y_test', y_test)):
            if not np.array_equal(saved[key], value):
                return False
        return all(name in saved['variants'] and 'metric_ci' in saved['variants'][name] for name in VARIANTS)
    except Exception:
        return False


def run_dataset(dataset, options):
    torch.set_num_threads(1)
    size = REFERENCE_SIZES[dataset]
    try:
        args = get_config('performance')
        args['sample_sizes'] = None
        x, y = load_data(dataset, args, raw=True)
        test_idx, splits = make_sample_splits(dataset, x, y, args)
        train_idx = splits[size]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx].to_numpy()
        reference_checked = check_reference(dataset, size, train_idx, test_idx, y_test, args)
        params = load_configured_params(dataset, 'kan_gam', size, args)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result_path = OUTPUT_DIR / f'{dataset}.pkl'
        if result_path.exists() and not options.overwrite:
            if valid_existing(result_path, dataset, size, train_idx, test_idx, y_test, params, options, args):
                log(dataset, 'Reusing existing completed result')
                return dataset, 'reused', ''
            raise ValueError(f'Existing result is incompatible or incomplete: {result_path}; inspect it or use --overwrite')
        x_train, x_test, preprocessing = preprocess_split(x.iloc[train_idx], x.iloc[test_idx])
        fit_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=0)
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory(prefix='symbolic-kaam-') as work_dir:
            try:
                os.chdir(work_dir)
                log(dataset, f'N={size}: training KAAM (1/4)')
                _, model, _, training_time = fit_with_saved_params('kan_gam', dataset, x_train, y_train, args, size)
                classes = np.asarray(model.classes_).reshape(-1)
                original_prob = np.asarray(model.predict_proba(x_test))
                original_pred = classes[np.argmax(original_prob, axis=1)]
                log(dataset, 'Selecting symbolic functions on training data (2/4)')
                start = time.perf_counter()
                model.model(model.dataset['train_input'])
                model.model.auto_symbolic()
                symbolic_time = time.perf_counter() - start
                symbolic_prob = np.asarray(model.predict_proba(x_test))
                symbolic_pred = classes[np.argmax(symbolic_prob, axis=1)]
                # Capture logits before extracting the formula, which may change cached activations.
                with torch.no_grad():
                    symbolic_logits = model.model(torch.as_tensor(x_test.to_numpy(), dtype=torch.float32, device=model.device)).detach().cpu().numpy()
                formulas, symbols = model.model.symbolic_formula()
                if len(formulas) != len(classes):
                    raise ValueError('The number of symbolic logits differs from the number of classes')
                log(dataset, 'Checking extracted formula and evaluating two-decimal formula (3/4)')
                formula_logits = evaluate_formulas(formulas, symbols, x_test)
                max_logit_error = float(np.max(np.abs(formula_logits - symbolic_logits)))
                if not np.allclose(formula_logits, symbolic_logits, atol=options.formula_atol, rtol=options.formula_rtol):
                    raise ValueError(f'Extracted formula does not reproduce symbolic model logits: max difference {max_logit_error:g}')
                rounded = [ex_round(formula, 2) for formula in formulas]
                rounding_fallback = False
                rounding_failure = None
                try:
                    rounded_logits = evaluate_formulas(rounded, symbols, x_test)
                except ValueError as exc:
                    if str(exc) != 'Symbolic expressions produced non-finite logits on test':
                        raise
                    rounding_fallback = True
                    rounding_failure = str(exc)
                    rounded_prob = symbolic_prob.copy()
                    log(dataset, 'Two-decimal formula produced non-finite logits; using unrounded symbolic probabilities for all test rows')
                else:
                    rounded_prob = softmax(rounded_logits, axis=1)
                predictions = {'original': (original_pred, original_prob),
                               'formula': (symbolic_pred, symbolic_prob),
                               'formula_2dec': (classes[np.argmax(rounded_prob, axis=1)], rounded_prob)}
                for name, (pred, proba) in predictions.items():
                    get_metrics(y_test, pred, proba, classes=classes, n_bins=args['ece_bins'])
                log(dataset, f'Calculating {options.n_bootstrap} paired bootstrap replicates (4/4)')
                summaries, differences, n_valid = paired_bootstrap(y_test, classes, predictions, options.n_bootstrap, options.seed, args['ece_bins'])
                for name in VARIANTS:
                    summaries[name]['y_pred'], summaries[name]['y_proba'] = predictions[name]
                result = {'dataset': dataset, 'model': 'kan_gam', 'train_size': size,
                          'train_indices': train_idx, 'fit_indices': fit_idx, 'validation_indices': val_idx,
                          'test_indices': test_idx, 'y_test': y_test, 'classes': classes,
                          'hyperparameters': params, 'split_seed': args['split_seed'], 'preprocessing': preprocessing,
                          'features': list(x_train.columns), 'symbols': [str(s) for s in symbols],
                          'formulas': [str(f) for f in formulas], 'formulas_rounded': [str(f) for f in rounded],
                          'round_decimals': 2, 'training_time': training_time, 'symbolic_time': symbolic_time,
                          'rounding_fallback': rounding_fallback, 'rounding_failure': rounding_failure,
                          'reference_split_checked': reference_checked, 'formula_max_logit_error': max_logit_error,
                          'formula_atol': options.formula_atol, 'formula_rtol': options.formula_rtol,
                          'variants': summaries, 'paired_differences': differences,
                          'bootstrap': {'n_bootstrap': options.n_bootstrap, 'n_valid': n_valid, 'ci': 95,
                                        'method': 'percentile', 'seed': options.seed, 'ece_bins': args['ece_bins']}}
            finally:
                os.chdir(original_dir)
        fd, temp_path = tempfile.mkstemp(prefix=f'.{dataset}-', suffix='.tmp', dir=OUTPUT_DIR)
        try:
            with os.fdopen(fd, 'wb') as file:
                pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, result_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        log(dataset, f'Saved {result_path}; valid bootstrap replicates: {n_valid}/{options.n_bootstrap}')
        return dataset, 'completed', ''
    except Exception:
        error = traceback.format_exc()
        log(dataset, f'FAILED\n{error}')
        return dataset, 'failed', error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=1, help='Number of datasets processed in parallel')
    parser.add_argument('--datasets', nargs='+', choices=REFERENCE_SIZES, default=list(REFERENCE_SIZES))
    parser.add_argument('--n-bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0, help='Bootstrap seed')
    parser.add_argument('--formula-atol', type=float, default=1e-4)
    parser.add_argument('--formula-rtol', type=float, default=1e-4)
    parser.add_argument('--overwrite', action='store_true', help='Replace existing results after successful completion')
    options = parser.parse_args()
    if options.jobs < 1 or options.n_bootstrap < 1 or options.formula_atol < 0 or options.formula_rtol < 0:
        parser.error('Jobs and bootstrap replicates must be positive; tolerances must be nonnegative')
    if len(set(options.datasets)) != len(options.datasets):
        parser.error('Each dataset can only be selected once')
    print(f'Total datasets: {len(options.datasets)} | parallel jobs: {options.jobs}', flush=True)
    counts = {'completed': 0, 'reused': 0, 'failed': 0}
    with parallel_config(backend='loky', inner_max_num_threads=1):
        finished = Parallel(n_jobs=options.jobs, return_as='generator_unordered')(
            delayed(run_dataset)(dataset, options) for dataset in options.datasets)
        for number, (dataset, status, _) in enumerate(finished, 1):
            counts[status] += 1
            print(f"Progress: {number}/{len(options.datasets)} | completed={counts['completed']} reused={counts['reused']} failed={counts['failed']} | {dataset}: {status}", flush=True)
    if counts['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
