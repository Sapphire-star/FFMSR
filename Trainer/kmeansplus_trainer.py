import os
from logging import getLogger
from time import time
from datetime import date
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from sklearn.cluster import KMeans
from torch.nn import functional as F
from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    EvaluatorType, KGDataLoaderState, get_tensorboard, set_color, get_gpu_usage, WandbLogger

class KMEANS:
    def __init__(self, n_clusters=30, max_iter=300, verbose=True, device=torch.device('cpu'), init='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = device
        self.init = init

        self.labels = None
        self.centers = None

    def _initialize_centers(self, x):
        if self.init == 'random':
            indices = torch.randint(0, x.shape[0], (self.n_clusters,))
        elif self.init == 'kmeans++':
            indices = [int(np.random.choice(range(x.shape[0]), 1))]
            for _ in range(1, self.n_clusters):
                dist_sq = torch.min(torch.cdist(x[indices], x, p=2) ** 2, dim=0)[0]
                prob = dist_sq / dist_sq.sum()
                indices.append(int(np.random.choice(range(x.shape[0]), 1, p=prob.cpu().numpy())))
        else:
            raise ValueError("Unsupported init method")
        self.centers = x[indices]

    def _assign_clusters(self, x):
        distances = torch.cdist(x, self.centers, p=2)
        self.labels = torch.argmin(distances, dim=1)

    def _update_centers(self, x):
        centers = torch.stack([x[self.labels == i].mean(0) for i in range(self.n_clusters)])
        self.centers = centers

    def fit(self, x):
        self._initialize_centers(x)
        for iteration in range(self.max_iter):
            centers_old = self.centers.clone()
            self._assign_clusters(x)
            self._update_centers(x)
            center_shift = torch.sum(torch.sqrt(torch.sum((centers_old - self.centers) ** 2, dim=1)))

            if self.verbose:
                print(f'Iteration {iteration + 1}, center shift {center_shift.item():.4f}.')
            if center_shift < 1e-4:
                break

    def predict(self, x):
        if self.centers is None:
            raise RuntimeError("Model not trained")
        distances = torch.cdist(x, self.centers, p=2)
        return torch.argmin(distances, dim=1)


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config_A, config_B, model_A, model_B):
        self.config_A = config_A
        self.config_B = config_B
        self.model_A = model_A
        self.model_B = model_B

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config_A, config_B, model_A, model_B):
        super(Trainer, self).__init__(config_A, config_B, model_A, model_B)

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.learner = config_A['learner']
        self.learning_rate = config_A['learning_rate']
        self.epochs = config_A['epochs']
        self.eval_step = min(config_A['eval_step'], self.epochs)
        self.stopping_step = config_A['stopping_step']
        self.clip_grad_norm = config_A['clip_grad_norm']
        self.valid_metric = config_A['valid_metric'].lower()
        self.valid_metric_bigger = config_A['valid_metric_bigger']
        self.test_batch_size = config_A['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config_A['use_gpu']
        self.device = config_A['device']
        self.checkpoint_dir = config_A['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        self.weight_decay = config_A['weight_decay']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict_A = dict()
        self.train_loss_dict_B = dict()
        self.optimizer_A = self._build_optimizer_A()
        self.optimizer_B = self._build_optimizer_B()

    def fit(self, train_data):
        pass

    def evaluate(self, eval_data):
        pass

    def _build_optimizer_A(self, **kwargs):
        params_A = kwargs.pop('params_A', self.model_A.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if self.config_A['reg_weight'] and weight_decay and weight_decay * self.config_A['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params_A, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params_A, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params_A, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params_A, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params_A, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params_A, lr=learning_rate)
        return optimizer

    def _build_optimizer_B(self, **kwargs):
        params_B = kwargs.pop('params_B', self.model_B.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if self.config_B['reg_weight'] and weight_decay and weight_decay * self.config_B['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params_B, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params_B, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params_B, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params_B, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params_B, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning('Sparse Bdam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Bdam optimizer')
            optimizer = optim.Adam(params_B, lr=learning_rate)
        return optimizer

    def _train_epoch_A(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model_A.train()
        loss_func = loss_func or self.model_A.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train_A {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer_A.zero_grad()
            losses = self.model_A.calculate_loss(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model_A.parameters(), **self.clip_grad_norm)
            self.optimizer_A.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('Loss: ' + str(losses.item()), 'yellow'))
        with torch.no_grad():
            prompt = self.model_A.get_fed_emb().clone().detach()
        return total_loss, prompt

    def _train_epoch_B(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model_B.train()
        loss_func = loss_func or self.model_B.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train_B {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer_B.zero_grad()
            losses = self.model_B.calculate_loss(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model_B.parameters(), **self.clip_grad_norm)
            self.optimizer_B.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('Loss: ' + str(losses.item()), 'yellow'))
        with torch.no_grad():
            prompt = self.model_B.get_fed_emb().clone().detach()
        return total_loss, prompt

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output_A(self, epoch_idx, s_time, e_time, losses):
        des = self.config_A['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss_A%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss_A', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _generate_train_loss_output_B(self, epoch_idx, s_time, e_time, losses):
        des = self.config_B['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss_B%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss_B', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)



class FedtrainTrainer(Trainer):
    r"""PretrainTrainer is designed for pre-training.
    It can be inherited by the trainer which needs pre-training and fine-tuning.
    """

    def __init__(self, config_A, config_B, model_A, model_B):
        super(FedtrainTrainer, self).__init__(config_A, config_B, model_A, model_B)
        self.pretrain_epochs = self.config_A['pretrain_epochs']
        self.save_step = self.config_A['save_step']
        self.k = config_A['cluster_centroids']
        self.iters = config_A['cluster_iters']
        self.kmeans = KMEANS(n_clusters=self.k, max_iter=self.iters, verbose=True, device=self.device)
        # self.attn_layer = nn.MultiheadAttention(config_A['hidden_size'], num_heads=4).to(config_A['device'])

    def save_pretrained_model_A(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            'config': self.config_A,
            'epoch': epoch,
            'state_dict': self.model_A.state_dict(),
            'optimizer': self.optimizer_A.state_dict(),
            'other_parameter': self.model_A.other_parameter(),
        }
        torch.save(state, saved_model_file)

    def save_pretrained_model_B(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            'config': self.config_B,
            'epoch': epoch,
            'state_dict': self.model_B.state_dict(),
            'optimizer': self.optimizer_B.state_dict(),
            'other_parameter': self.model_B.other_parameter(),
        }
        torch.save(state, saved_model_file)

    def fedtrain(self, train_data_A, train_data_B, verbose=True, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss_A, item_prompt_A = self._train_epoch_A(train_data_A, epoch_idx, show_progress=show_progress)
            train_loss_B, item_prompt_B = self._train_epoch_B(train_data_B, epoch_idx, show_progress=show_progress)
            self.train_loss_dict_A[epoch_idx] = sum(train_loss_A) if isinstance(train_loss_A, tuple) else train_loss_A
            self.train_loss_dict_B[epoch_idx] = sum(train_loss_B) if isinstance(train_loss_B, tuple) else train_loss_B
            training_end_time = time()
            train_loss_output_A = \
                self._generate_train_loss_output_A(epoch_idx, training_start_time, training_end_time, train_loss_A)
            train_loss_output_B = \
                self._generate_train_loss_output_B(epoch_idx, training_start_time, training_end_time, train_loss_B)
            if verbose:
                self.logger.info(train_loss_output_A)
                self.logger.info(train_loss_output_B)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss_A)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss_B)
            item_prompt_A = item_prompt_A[1:]
            item_prompt_B = item_prompt_B[1:]
            all_prompt = torch.cat((item_prompt_A, item_prompt_B), dim=0)
            self.kmeans.fit(all_prompt)
            clustered_vectors = []
            labels = self.kmeans.labels
            for cluster_idx in range(self.k):
                cluster_vectors = all_prompt[labels == cluster_idx]
                clustered_vectors.append(cluster_vectors)
            aggregate_vectors = []
            for cluster_vectors in clustered_vectors:
                aggregate_vector = torch.mean(cluster_vectors, dim=0)
                aggregate_vectors.append(aggregate_vector)
            cluster_prompt = torch.zeros_like(all_prompt)
            for i in range(all_prompt.size(0)):
                cluster_prompt[i] = aggregate_vectors[labels[i]]
            cluster_prompt_A = cluster_prompt[:item_prompt_A.shape[0]].to(self.device)
            cluster_prompt_B = cluster_prompt[item_prompt_A.shape[0]:].to(self.device)
            self.model_A.global_prompt.data[1:] = cluster_prompt_A
            self.model_B.global_prompt.data[1:] = cluster_prompt_B
            if (epoch_idx + 1) % self.save_step == 0:
                saved_model_file_A = os.path.join(
                    self.checkpoint_dir,
                    '{}-{}-{}-{}.pth'.format(self.config_A['model'], self.config_A['dataset'], str(epoch_idx + 1), str(date.today()))
                )
                saved_model_file_B = os.path.join(
                    self.checkpoint_dir,
                    '{}-{}-{}-{}.pth'.format(self.config_B['model'], self.config_B['dataset'], str(epoch_idx + 1), str(date.today()))
                )
                self.save_pretrained_model_A(epoch_idx, saved_model_file_A)
                self.save_pretrained_model_B(epoch_idx, saved_model_file_B)
                update_output_A = set_color('Saving current', 'blue') + ': %s' % saved_model_file_A
                update_output_B = set_color('Saving current', 'blue') + ': %s' % saved_model_file_B
                if verbose:
                    self.logger.info(update_output_A)
                    self.logger.info(update_output_B)

        return self.best_valid_score, self.best_valid_result