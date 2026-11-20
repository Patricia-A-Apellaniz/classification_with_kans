import os
import scipy
import torch
import random

import numpy as np
import pandas as pd

from typing import Callable
from numpy.typing import ArrayLike
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
from src.models.nam.data import NAMDataset
from src.models.nam.trainer import Trainer
from src.models.nam.models import get_num_units
from src.models.nam.models import NAM, MultiTaskNAM
from src.models.nam.models.saver import Checkpointer
from src.models.nam.trainer.losses import (
    make_penalized_loss_func,
    make_multiclass_penalized_loss_func,
)


class NAMBase:
    def __init__(
            self,
            units_multiplier: int = 2,
            num_basis_functions: int = 64,
            hidden_sizes: list = [64, 32],
            dropout: float = 0.1,
            feature_dropout: float = 0.05,
            batch_size: int = 1024,
            num_workers: int = 0,
            num_epochs: int = 1000,
            log_dir: str = None,
            val_split: float = 0.15,
            device: str = 'cpu',
            lr: float = 0.02082,
            decay_rate: float = 0.0,
            output_reg: float = 0.2078,
            l2_reg: float = 0.0,
            save_model_frequency: int = 10,
            patience: int = 60,
            monitor_loss: bool = True,
            early_stop_mode: str = 'min',
            loss_func: Callable = None,
            metric: str = None,
            num_learners: int = 1,
            n_jobs: int = None,
            warm_start: bool = False,
            random_state: int = 42
    ) -> None:
        self.units_multiplier = units_multiplier
        self.num_basis_functions = num_basis_functions
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.val_split = val_split
        self.device = device
        self.lr = lr
        self.decay_rate = decay_rate
        self.output_reg = output_reg
        self.l2_reg = l2_reg
        self.save_model_frequency = save_model_frequency
        self.patience = patience
        self.monitor_loss = monitor_loss
        self.early_stop_mode = early_stop_mode
        self.loss_func = loss_func
        self.metric = metric
        self.num_learners = num_learners
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.random_state = random_state

        self._best_checkpoint_suffix = 'best'
        self._fitted = False

    def _set_random_state(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        return

    def _initialize_models(self, X, y):
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.num_inputs = X.shape[1]
        self.models = []
        for _ in range(self.num_learners):
            model = NAM(num_inputs=self.num_inputs,
                        num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                        dropout=self.dropout,
                        feature_dropout=self.feature_dropout,
                        hidden_sizes=self.hidden_sizes)
            self.models.append(model)

        return

    def _models_to_device(self, device):
        for model in self.models:
            model.to(device)

        return

    def fit(self, X, y, w=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        if isinstance(w, (pd.DataFrame, pd.Series)):
            w = w.to_numpy()

        self._set_random_state()
        if not self.warm_start or not self._fitted:
            self._initialize_models(X, y)

        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, w=None) -> None:
        self._models_to_device(self.device)

        # self._preprocessor = MinMaxScaler(feature_range = (-1, 1))

        # dataset = NAMDataset(self._preprocessor.fit_transform(X), y, w)
        dataset = NAMDataset(X, y, w)

        if getattr(self, '_multiclass', False):
            if self.loss_func is not None:
                raise ValueError(
                    'A custom loss function is not supported for '
                    'multiclass NAM in this implementation.'
                )

            self.criterion = make_multiclass_penalized_loss_func(
                self.output_reg,
                self.l2_reg,
            )

            trainer_metric = None
            trainer_monitor_loss = True
            trainer_early_stop_mode = 'min'
        else:
            # Preserve the current binary and regression behavior.
            self.criterion = make_penalized_loss_func(
                self.loss_func,
                self.regression,
                self.output_reg,
                self.l2_reg,
            )

            trainer_metric = self.metric
            trainer_monitor_loss = self.monitor_loss
            trainer_early_stop_mode = self.early_stop_mode

        self.trainer = Trainer(
            models=self.models,
            dataset=dataset,
            metric=trainer_metric,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_epochs=self.num_epochs,
            log_dir=self.log_dir,
            val_split=self.val_split,
            test_split=None,
            device=self.device,
            lr=self.lr,
            decay_rate=self.decay_rate,
            save_model_frequency=self.save_model_frequency,
            patience=self.patience,
            monitor_loss=trainer_monitor_loss,
            early_stop_mode=trainer_early_stop_mode,
            criterion=self.criterion,
            regression=self.regression,
            num_learners=self.num_learners,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        self.trainer.train_ensemble()
        self.trainer.close()

        # Move models to cpu so predictions can be made on cpu data
        self._models_to_device('cpu')

        self._fitted = True
        return self

    def predict(self, X) -> ArrayLike:
        if not self._fitted:
            raise NotFittedError('''This NAM instance is not fitted yet. Call \'fit\' 
                with appropriate arguments before using this method.''')

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # X = self._preprocessor.transform(X)
        X = torch.tensor(X, requires_grad=False, dtype=torch.float)
        predictions = np.zeros((X.shape[0], self.num_tasks))

        for model in self.models:
            preds, _ = model.forward(X)
            predictions += preds.detach().cpu().numpy()

        # predictions = self._preprocessor.inverse_transform(predictions)
        return predictions / self.num_learners

    def plot(self, feature_index) -> None:
        num_samples = 1000
        X = np.zeros((num_samples, self.num_inputs))
        X[:, feature_index] = np.linspace(-1.0, 1.0, num_samples)

        feature_outputs = []
        for model in self.models:
            # (examples, tasks, features)
            _, fnns_out = model.forward(torch.tensor(X, dtype=torch.float32))
            if self.num_tasks == 1:
                fnns_out = fnns_out.unsqueeze(dim=1)
            # (examples, tasks)
            feature_outputs.append(fnns_out[:, :, feature_index].detach().cpu().numpy())

        # (learners, examples, tasks)
        feature_outputs = np.stack(feature_outputs, axis=0)
        # (examples, tasks)
        y = np.mean(feature_outputs, axis=0).squeeze()
        conf_int = np.std(feature_outputs, axis=0).squeeze()
        # TODO: Scale conf_int according to units of y

        # X = self._preprocessor.inverse_transform(X)

        return {'x': X[:, feature_index], 'y': y, 'conf_int': conf_int}

    def load_checkpoints(self, checkpoint_dir):
        self.models = []
        for i in range(self.num_learners):
            checkpointer = Checkpointer(os.path.join(checkpoint_dir, str(i)))
            model = checkpointer.load(self._best_checkpoint_suffix)
            model.eval()
            self.num_tasks = 1 if isinstance(model, NAM) else model.num_tasks
            self.models.append(model)

        self._fitted = True
        return


class NAMClassifier(NAMBase):
    def __init__(
            self,
            units_multiplier: int = 2,
            num_basis_functions: int = 64,
            hidden_sizes: list = [64, 32],
            dropout: float = 0.1,
            feature_dropout: float = 0.05,
            batch_size: int = 1024,
            num_workers: int = 0,
            num_epochs: int = 1000,
            log_dir: str = None,
            val_split: float = 0.15,
            device: str = 'cpu',
            lr: float = 0.02082,
            decay_rate: float = 0.0,
            output_reg: float = 0.2078,
            l2_reg: float = 0.0,
            save_model_frequency: int = 10,
            patience: int = 60,
            monitor_loss: bool = True,
            early_stop_mode: str = 'min',
            loss_func: Callable = None,
            metric: str = None,
            num_learners: int = 1,
            n_jobs: int = None,
            warm_start: bool = False,
            random_state: int = 42
    ) -> None:
        super(NAMClassifier, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state
        )
        self.regression = False

    def _initialize_models(self, X, y):
        if not self._multiclass:
            # Use the original initialization for binary classification.
            return super()._initialize_models(X, y)

        self.num_tasks = len(self.classes_)
        self.num_inputs = X.shape[1]
        self.models = []

        for _ in range(self.num_learners):
            model = NAM(
                num_inputs=self.num_inputs,
                num_units=get_num_units(
                    self.units_multiplier,
                    self.num_basis_functions,
                    X,
                ),
                dropout=self.dropout,
                feature_dropout=self.feature_dropout,
                hidden_sizes=self.hidden_sizes,
                num_classes=self.num_tasks,
            )
            self.models.append(model)

    def fit(self, X, y, w=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        if isinstance(w, (pd.DataFrame, pd.Series)):
            w = w.to_numpy()

        y = np.asarray(y).reshape(-1)

        if np.isnan(y).any():
            raise ValueError(
                'NAM classification targets cannot contain missing values.'
            )

        self.classes_, encoded_y = np.unique(
            y,
            return_inverse=True,
        )

        if len(self.classes_) < 2:
            raise ValueError(
                'NAM classification requires at least two classes.'
            )

        self._multiclass = len(self.classes_) > 2

        if self._multiclass:
            # Cross-entropy requires class indices from 0 to K-1.
            return super().fit(X, encoded_y, w)

        # Preserve the existing binary labels and training path.
        return super().fit(X, y, w)

    def predict_proba(self, X) -> ArrayLike:
        if not self._multiclass:
            # Preserve the existing binary prediction path.
            return scipy.special.expit(super().predict(X))

        # Checkpoints reconstructed by PyTorch can start in training mode.
        # Disable dropout for multiclass inference.
        for model in self.models:
            model.eval()

        # NAMBase.predict averages the learners' logits.
        mean_logits = super().predict(X)

        return scipy.special.softmax(mean_logits, axis=1)

    def predict(self, X) -> ArrayLike:
        if not self._multiclass:
            return self.predict_proba(X).round()

        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class NAMRegressor(NAMBase):
    def __init__(
            self,
            units_multiplier: int = 2,
            num_basis_functions: int = 64,
            hidden_sizes: list = [64, 32],
            dropout: float = 0.1,
            feature_dropout: float = 0.05,
            batch_size: int = 1024,
            num_workers: int = 0,
            num_epochs: int = 1000,
            log_dir: str = None,
            val_split: float = 0.15,
            device: str = 'cpu',
            lr: float = 0.02082,
            decay_rate: float = 0.0,
            output_reg: float = 0.2078,
            l2_reg: float = 0.0,
            save_model_frequency: int = 10,
            patience: int = 60,
            monitor_loss: bool = True,
            early_stop_mode: str = 'min',
            loss_func: Callable = None,
            metric: str = None,
            num_learners: int = 1,
            n_jobs: int = None,
            warm_start: bool = False,
            random_state: int = 42
    ) -> None:
        super(NAMRegressor, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state
        )
        self.regression = True


class MultiTaskNAMClassifier(NAMClassifier):
    def __init__(
            self,
            units_multiplier: int = 2,
            num_basis_functions: int = 64,
            hidden_sizes: list = [64, 32],
            num_subnets: int = 2,
            dropout: float = 0.1,
            feature_dropout: float = 0.05,
            batch_size: int = 1024,
            num_workers: int = 0,
            num_epochs: int = 1000,
            log_dir: str = None,
            val_split: float = 0.15,
            device: str = 'cpu',
            lr: float = 0.02082,
            decay_rate: float = 0.0,
            output_reg: float = 0.2078,
            l2_reg: float = 0.0,
            save_model_frequency: int = 10,
            patience: int = 60,
            monitor_loss: bool = True,
            early_stop_mode: str = 'min',
            loss_func: Callable = None,
            metric: str = None,
            num_learners: int = 1,
            n_jobs: int = None,
            warm_start: bool = False,
            random_state: int = 42
    ) -> None:
        super(MultiTaskNAMClassifier, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state
        )
        self.num_subnets = num_subnets

    def _initialize_models(self, X, y):
        self.num_inputs = X.shape[1]
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.models = []
        for _ in range(self.num_learners):
            model = MultiTaskNAM(num_inputs=X.shape[1],
                                 num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                                 num_subnets=self.num_subnets,
                                 num_tasks=y.shape[1],
                                 dropout=self.dropout,
                                 feature_dropout=self.feature_dropout,
                                 hidden_sizes=self.hidden_sizes)
            model.to(self.device)
            self.models.append(model)


class MultiTaskNAMRegressor(NAMRegressor):
    def __init__(
            self,
            units_multiplier: int = 2,
            num_basis_functions: int = 64,
            hidden_sizes: list = [64, 32],
            num_subnets: int = 2,
            dropout: float = 0.1,
            feature_dropout: float = 0.05,
            batch_size: int = 1024,
            num_workers: int = 0,
            num_epochs: int = 1000,
            log_dir: str = None,
            val_split: float = 0.15,
            device: str = 'cpu',
            lr: float = 0.02082,
            decay_rate: float = 0.995,
            output_reg: float = 0.2078,
            l2_reg: float = 0.0,
            save_model_frequency: int = 10,
            patience: int = 60,
            monitor_loss: bool = True,
            early_stop_mode: str = 'min',
            loss_func: Callable = None,
            metric: str = None,
            num_learners: int = 1,
            n_jobs: int = None,
            warm_start: bool = False,
            random_state: int = 42
    ) -> None:
        super(MultiTaskNAMRegressor, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners,
            n_jobs=n_jobs,
            warm_start=warm_start,
            random_state=random_state
        )
        self.num_subnets = num_subnets

    def _initialize_models(self, X, y):
        self.num_inputs = X.shape[1]
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.models = []
        for _ in range(self.num_learners):
            model = MultiTaskNAM(num_inputs=X.shape[1],
                                 num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                                 num_subnets=self.num_subnets,
                                 num_tasks=y.shape[1],
                                 dropout=self.dropout,
                                 feature_dropout=self.feature_dropout,
                                 hidden_sizes=self.hidden_sizes)
            model.to(self.device)
            self.models.append(model)
