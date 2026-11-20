import argparse
import pickle
from pathlib import Path

from src.models.model_utils import add_bootstrap_metrics


parser = argparse.ArgumentParser()

parser.add_argument(
    'input_path',
    type=Path,
    help='Existing result pickle with saved test predictions.',
)

parser.add_argument(
    'output_path',
    type=Path,
    help='New result pickle; the input is not overwritten.',
)

args = parser.parse_args()

with args.input_path.open('rb') as f:
    result = pickle.load(f)

previous_bootstrap = result.get('bootstrap', {})

add_bootstrap_metrics(
    result,
    n_bootstrap=previous_bootstrap.get('n_bootstrap', 1000),
    ci=previous_bootstrap.get('ci', 95),
    seed=previous_bootstrap.get('seed', 0),
    n_bins=10,
)

args.output_path.parent.mkdir(parents=True, exist_ok=True)

with args.output_path.open('xb') as f:
    pickle.dump(
        result,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print(f'Saved recalculated metrics to {args.output_path}')
