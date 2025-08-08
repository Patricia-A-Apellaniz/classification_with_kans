# Author: Juan Parras & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 05/08/2025

# Package imports
from kan import *
from src.models.nam.wrapper import NAMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#--------------------------------------------------------#
#                        NAM MODEL                       #
#--------------------------------------------------------#
class NAMModel:
    """
    Class for NAM model, it includes the definition of the model and functions to run the model
    """

    def __init__(self, num_epochs=1000, num_learners=20, metric='aucroc', early_stop_mode='max', n_jobs=1,
                 random_state=0,
                 num_basis_functions=64, hidden_size=[64, 32]):
        # The next two values are needed to obtain the optimal parameters using Sklearn's GridSearchCV
        self._estimator_type = 'classifier'
        self.classes_ = np.array([0, 1]).astype(int)

        self.hyperparameters = {'num_epochs': num_epochs,
                                'num_learners': num_learners,
                                'metric': metric,
                                'early_stop_mode': early_stop_mode,
                                'n_jobs': n_jobs,
                                'random_state': random_state,
                                'num_basis_functions': num_basis_functions,
                                'hidden_size': hidden_size}

    def fit(self, x, y):
        self.set_model()
        self.model.fit(x, y)
        return {}

    def set_model(self):
        self.model = NAMClassifier(num_epochs=self.hyperparameters["num_epochs"],
                                   num_learners=self.hyperparameters["num_learners"],
                                   metric=self.hyperparameters["metric"],
                                   early_stop_mode=self.hyperparameters["early_stop_mode"],
                                   n_jobs=self.hyperparameters["n_jobs"],
                                   random_state=self.hyperparameters["random_state"],
                                   num_basis_functions=self.hyperparameters["num_basis_functions"],
                                   hidden_sizes=self.hyperparameters["hidden_size"])

    def get_params(self, deep=False):
        return self.hyperparameters

    def set_params(self, num_epochs=1000, num_learners=20, metric='aucroc', early_stop_mode='max', n_jobs=1,
                   random_state=0, num_basis_functions=64, hidden_size=[64, 32]):
        self.hyperparameters = {'num_epochs': num_epochs,
                                'num_learners': num_learners,
                                'metric': metric,
                                'early_stop_mode': early_stop_mode,
                                'n_jobs': n_jobs,
                                'random_state': random_state,
                                'num_basis_functions': num_basis_functions,
                                'hidden_size': hidden_size}
        return self

    def predict_proba(self, data):
        prob_of_ones = self.model.predict_proba(data)
        prob_of_zeros = 1 - prob_of_ones
        return np.hstack([prob_of_zeros, prob_of_ones])

    def predict(self, data, threshold=0.5):
        proba_preds = self.predict_proba(data)
        return np.argmax(proba_preds, axis=1).astype(int)

    def run_model(self, x_train, x_test, y_train, y_test):
        self.set_model()

        # Train the model
        _ = self.model.fit(x_train, y_train)

        return {'model': self.model,
                'y_pred_proba': np.squeeze(self.predict_proba(x_test)),
                'y_pred': np.squeeze(self.predict(x_test))}


#--------------------------------------------------------#
#                         RF MODEL                       #
#--------------------------------------------------------#
class RandomForestModel:
    """
    Class for random forest model, it includes the definition of the model and functions to run the model
    """

    def __init__(self, n_estimators=20, criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1,
                 class_weight='balanced', max_features='log2', bootstrap=True, random_state=0, n_jobs=1):
        self.classification_flag = True

        # The next two values are needed to obtain the optimal parameters using Sklearn's GridSearchCV
        self._estimator_type = 'classifier'
        self.classes_ = np.array([0, 1]).astype(int)

        self.hyperparameters = {'n_estimators': n_estimators,
                                'criterion': criterion,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'class_weight': class_weight,
                                'max_features': max_features,
                                'bootstrap': bootstrap,
                                'random_state': random_state,
                                'n_jobs': n_jobs}
        self.model = RandomForestClassifier(**self.hyperparameters)

    def fit(self, x, y):
        self.classes_ = np.arange(len(np.unique(y))).astype(int)
        return self.model.fit(x, y)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def predict(self, x):
        return self.model.predict(x)

    def get_params(self, deep=False):
        return self.hyperparameters

    def set_params(self, n_estimators=20, criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1,
                   class_weight='balanced', max_features='log2', bootstrap=True, random_state=0, n_jobs=1):
        self.hyperparameters = {'n_estimators': n_estimators,
                                'criterion': criterion,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'class_weight': class_weight,
                                'max_features': max_features,
                                'bootstrap': bootstrap,
                                'random_state': random_state,
                                'n_jobs': n_jobs}
        self.model = RandomForestClassifier(**self.hyperparameters)
        return self

    def run_model(self, x_train, y_train, x_test, y_test):
        """
        Function to run the model
        :return:
        """
        # Train model
        y_all = np.concatenate([y_train, y_test])
        self.classes_ = np.arange(len(np.unique(y_all))).astype(int)
        _ = self.model.fit(x_train, y_train)

        return {'model': self.model,
                'y_pred_proba': np.squeeze(self.model.predict_proba(x_test)),
                'y_pred': np.squeeze(self.model.predict(x_test))}


#--------------------------------------------------------#
#                        KAN MODEL                       #
#--------------------------------------------------------#
class Kan_model:
    """
    Class for MLP model, it includes the definition of the model and functions to run the model
    """

    def __init__(self, hidden_dim=0, batch_size=500, grid=1, k=1, seed=0, lr=0.01, early_stop=True, steps=10000,
                 lamb=0.1, lamb_entropy=0.1, weight=True, sparse_init=False, mult_kan=False, try_gpu=False):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and try_gpu else "cpu")  # Note that code may need adaptation to work properly on GPU (only tested in CPU so far)

        # The next two values are needed to obtain the optimal parameters using Sklearn's GridSearchCV
        self._estimator_type = 'classifier'
        self.classes_ = np.array([0, 1]).astype(int)
        self.n_classes = None

        self.hyperparameters = {'hidden_dim': hidden_dim,
                                'batch_size': batch_size,
                                'grid': grid,
                                'k': k,
                                'seed': seed,
                                'lr': lr,
                                'early_stop': early_stop,
                                'steps': steps,
                                'lamb': lamb,
                                'lamb_entropy': lamb_entropy,
                                'weight': weight,
                                'sparse_init': sparse_init,
                                'mult_kan': mult_kan}

    def seed_all(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def fit(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        r = self.run_model(x_train, x_test, y_train, y_test)
        return r

    def get_params(self, deep=False):
        return self.hyperparameters

    def set_params(self, hidden_dim=0, batch_size=500, grid=1, k=1, seed=0, lr=0.01, early_stop=True, steps=10000,
                   lamb=0.1, lamb_entropy=0.1, weight=True, sparse_init=False, mult_kan=False):
        self.hyperparameters = {'hidden_dim': hidden_dim,
                                'batch_size': batch_size,
                                'grid': grid,
                                'k': k,
                                'seed': seed,
                                'lr': lr,
                                'early_stop': early_stop,
                                'steps': steps,
                                'lamb': lamb,
                                'lamb_entropy': lamb_entropy,
                                'weight': weight,
                                'sparse_init': sparse_init,
                                'mult_kan': mult_kan}
        return self

    def set_model(self, x_train, x_test, y_train, y_test):
        self.seed_all(self.hyperparameters["seed"])
        y_all = np.concatenate([y_train, y_test])
        self.n_classes = len(np.unique(y_all))
        self.classes_ = np.arange(self.n_classes).astype(int)
        self.input_size = x_train.shape[1]

        if self.hyperparameters["hidden_dim"] == 0:
            self.width = [self.input_size, self.n_classes]
        elif isinstance(self.hyperparameters["hidden_dim"], int):
            self.width = [self.input_size, self.hyperparameters["hidden_dim"], self.n_classes]
        else:
            self.width = [self.input_size] + self.hyperparameters["hidden_dim"] + [self.n_classes]

        if self.hyperparameters['mult_kan']:  # Add multiplication nodes
            new_width = []
            for i in range(len(self.width)):
                if i == 0 or i == len(self.width) - 1:
                    new_width.append(self.width[i])
                else:
                    new_width.append([self.width[i], self.width[i]])
            self.width = new_width

        self.model = KAN(width=self.width, grid=self.hyperparameters["grid"], k=self.hyperparameters["k"],
                         seed=0, device=self.device, sparse_init=self.hyperparameters["sparse_init"])

        if isinstance(x_train, np.ndarray):
            self.x_train = torch.from_numpy(x_train).to(self.device).float()
            self.y_train = np.squeeze(torch.from_numpy(y_train).to(self.device)).long()
            self.x_test = torch.from_numpy(x_test).to(self.device).float()
            self.y_test = np.squeeze(torch.from_numpy(y_test).to(self.device)).long()
        else:
            self.x_train = torch.from_numpy(x_train.values).to(self.device).float()
            self.y_train = np.squeeze(torch.from_numpy(y_train.values).to(self.device)).long()
            self.x_test = torch.from_numpy(x_test.values).to(self.device).float()
            self.y_test = np.squeeze(torch.from_numpy(y_test.values).to(self.device)).long()

        # Note that KAN interface uses "test" for what we call "val": we reverse here for consistency
        self.dataset = {'train_input': self.x_train,
                        'train_label': self.y_train,
                        'test_input': self.x_test,
                        'test_label': self.y_test}

        if self.hyperparameters["weight"]:
            # Weight the classes according to the number of samples of each class to correct for class imbalance
            class_weigths = []
            freq_per_class = torch.bincount(self.y_train)
            for i in range(self.n_classes):
                class_weigths.append(len(self.y_train) / (self.n_classes * freq_per_class[i].item()))
            class_weights = torch.tensor(class_weigths).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def predict_proba(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device).float()
        elif isinstance(data, pd.DataFrame):
            data = torch.from_numpy(data.values).to(self.device).float()
        else:
            pass  # Assume it is already a tensor
        sm = nn.Softmax(dim=1)
        proba = sm(self.model.forward(data)).detach().cpu().numpy()
        return proba

    def predict(self, data, threshold=0.5):
        proba = self.predict_proba(data)
        return np.argmax(proba, axis=1).astype(int)

    def custom_fit(self, dataset, steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0.,
                   lamb_coefdiff=0., update_grid=True, grid_update_num=10, lr=1., start_grid_update_step=-1,
                   stop_grid_update_step=50, batch=-1,
                   save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1,
                   img_folder='./video', singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n',
                   display_metrics=None, early_stop=True, patience=30, verbose=1):

        if lamb > 0. and not self.model.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')

        old_save_act, old_symbolic_enabled = self.model.disable_symbolic_in_fit(lamb)

        if verbose > 0:
            pbar = tqdm(range(steps), desc='description', ncols=100)
        else:
            pbar = range(steps)

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        optimizer = torch.optim.Adam(self.model.get_params(), lr=lr)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        results['train_metrics'] = []
        results['test_metrics'] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
        else:
            batch_size = batch
        batch_size_test = dataset['test_input'].shape[0]

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        best_loss = np.inf
        patience_counter = 0

        for _ in pbar:

            if _ == steps - 1 and old_save_act:
                self.model.save_act = True

            if save_fig and _ % save_fig_freq == 0:
                save_act = self.model.save_act
                self.model.save_act = True

            n_batches_train = len(dataset['train_input']) // batch_size

            for ibt in range(n_batches_train):

                batch_start = ibt * batch_size
                batch_end = min((ibt + 1) * batch_size, len(dataset['train_input']))
                train_id = np.arange(batch_start, batch_end)

                if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                    self.model.update_grid(dataset['train_input'][train_id])

                pred_train = self.model.forward(dataset['train_input'][train_id],
                                                singularity_avoiding=singularity_avoiding, y_th=y_th)
                train_loss = self.criterion(pred_train, dataset['train_label'][train_id])
                if self.model.save_act:
                    if reg_metric == 'edge_backward':
                        self.model.attribute()
                    if reg_metric == 'node_backward':
                        self.model.node_attribute()
                    reg_ = self.model.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred_test = self.model.forward(dataset['test_input'])
            test_loss = self.criterion(pred_test, dataset['test_label'])

            # For conveniency, we get train loss and reg on the last batch only
            results['train_loss'].append(train_loss.cpu().detach().numpy())
            results['test_loss'].append(test_loss.cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if _ % log == 0 and verbose > 0:
                if display_metrics == None:
                    pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (
                        torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(),
                        reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)

            if save_fig and _ % save_fig_freq == 0:
                self.model.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_),
                                beta=beta)
                plt.savefig(img_folder + os.sep + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()
                self.model.save_act = save_act

            if early_stop:
                if results['test_loss'][-1] < best_loss:
                    best_loss = results['test_loss'][-1]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > patience:
                        print(f'Early stopping at step {_}')
                        break

        self.model.log_history('fit')
        # revert back to original state
        self.model.symbolic_enabled = old_symbolic_enabled
        return results

    def run_model(self, x_train, x_test, y_train, y_test):
        self.set_model(x_train, x_test, y_train, y_test)

        # Train the model
        _ = self.custom_fit(self.dataset, batch=self.hyperparameters["batch_size"],
                            steps=self.hyperparameters["steps"], lamb=self.hyperparameters["lamb"],
                            lamb_entropy=self.hyperparameters["lamb_entropy"], lr=self.hyperparameters["lr"],
                            early_stop=self.hyperparameters["early_stop"], patience=30,
                            save_fig=False, verbose=0)

        return {'model': self.model,
                'y_pred_proba': np.squeeze(self.predict_proba(self.x_test)),
                'y_pred': np.squeeze(self.predict(self.x_test))}

    def prune(self):
        self.model = self.model.prune()


#--------------------------------------------------------#
#                        LR MODEL                        #
#--------------------------------------------------------#
class LogisticRegressionModel:
    """
    Class for Logistic Regression model, including the definition and functions to train, validate, and test the model.
    """

    def __init__(self, C=0, penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=0):
        # The next two values are needed to obtain the optimal parameters using Sklearn's GridSearchCV
        self._estimator_type = 'classifier'
        self.classes_ = np.array([0, 1]).astype(int)

        self.hyperparameters = {'C': C,
                                'penalty': penalty,
                                'solver': solver,
                                'max_iter': max_iter,
                                'class_weight': class_weight,
                                'random_state': random_state}

        # Initialize the Logistic Regression model
        self.model = LogisticRegression(**self.hyperparameters)

    def fit(self, x, y):
        self.classes_ = np.arange(len(np.unique(y))).astype(int)
        return self.model.fit(x, y)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def predict(self, x):
        return self.model.predict(x)

    def get_params(self, deep=False):
        return self.hyperparameters

    def set_params(self, C=0, penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=0):
        self.hyperparameters = {'C': C,
                                'penalty': penalty,
                                'solver': solver,
                                'max_iter': max_iter,
                                'class_weight': class_weight,
                                'random_state': random_state}
        self.model = LogisticRegression(**self.hyperparameters)
        return self

    def run_model(self, x_train, y_train, x_test, y_test):
        """
        Train the Logistic Regression model using hyperparameter tuning, validate it, and test it.
        :return: Trained model, validation predictions, test predictions
        """
        # Train the model
        y_all = np.concatenate([y_train, y_test])
        self.classes_ = np.arange(len(np.unique(y_all))).astype(int)
        _ = self.model.fit(x_train, y_train)

        return {'model': self.model,
                'y_pred_proba': np.squeeze(self.model.predict_proba(x_test)),
                'y_pred': np.squeeze(self.model.predict(x_test))}

    def get_coefficients(self, model):
        """
        Retrieve coefficients and intercept of the trained logistic regression model.
        :param model: Trained logistic regression model
        :return: Coefficients and intercept as lists
        """
        coefficients = model.coef_.flatten().tolist()
        intercept = model.intercept_.tolist()[0]
        return coefficients, intercept


#--------------------------------------------------------#
#                        MLP MODEL                       #
#--------------------------------------------------------#
class Mlp_model:
    """
    Class for MLP model, it includes the definition of the model and functions to run the model
    """

    def __init__(self, hidden_layer_sizes=(32,), max_iter=10000, early_stopping=True, alpha=0.0001):
        self.classification_flag = True

        # The next two values are needed to obtain the optimal parameters using Sklearn's GridSearchCV
        self._estimator_type = 'classifier'
        self.classes_ = np.array([0, 1]).astype(int)

        self.hyperparameters = {'hidden_layer_sizes': hidden_layer_sizes,
                                'max_iter': max_iter,
                                'early_stopping': early_stopping,
                                'alpha': alpha}

        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                   max_iter=max_iter,
                                   early_stopping=early_stopping,
                                   alpha=alpha)

    def fit(self, x, y):
        self.classes_ = np.arange(len(np.unique(y))).astype(int)
        return self.model.fit(x, y)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def predict(self, x):
        return self.model.predict(x)

    def get_params(self, deep=False):
        return self.hyperparameters

    def set_params(self, hidden_layer_sizes=(32,), max_iter=10000, early_stopping=True, alpha=0.0001):
        self.hyperparameters = {'hidden_layer_sizes': hidden_layer_sizes,
                                'max_iter': max_iter,
                                'early_stopping': early_stopping,
                                'alpha': alpha}
        return self

    def run_model(self, x_train, y_train, x_test, y_test):
        """
        Function to run the model
        :return:
        """
        # Train model
        y_all = np.concatenate([y_train, y_test])
        self.classes_ = np.arange(len(np.unique(y_all))).astype(int)
        _ = self.model.fit(x_train, y_train)

        return {'model': self.model,
                'y_pred_proba': np.squeeze(self.model.predict_proba(x_test)),
                'y_pred': np.squeeze(self.model.predict(x_test))}
