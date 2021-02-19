import argparse
import collections
import json

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import dataset
import pytorch_lightning as pl
from domain_learner \
    import LearnerValueFunction, CharEncoding, PositionalEncoding, collate_concat, batched
from response_prediction import split_train_val_test
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
import matplotlib
from matplotlib import pyplot as plt
from cycler import cycler


class QuestionDifficultyDataset(torch.utils.data.Dataset):
    def __init__(self, path, min_observations=1):
        base = dataset.CognitiveTutorDataset(path)

        self.data = []

        for q, obs in base.obs_by_problem.items():
            if len(obs) >= min_observations:
                self.data.append((q, np.mean([o[1] for o in obs])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, np.float32(y)

class Featurizer:
    def embed(self, q):
        raise NotImplemented()

    def dimension(self):
        raise NotImplemented()

class HandcraftedFeaturizer(Featurizer):
    def __init__(self):
        self.nonlinear = True

    def dimension(self):
        return 5

    def embed(self, q):
        return np.array([len(q), q.count('+'), q.count('('), q.count('-'), q.count('/')])

class LSTMFeaturizer(Featurizer, nn.Module):
    def __init__(self):
        super().__init__()
        self.char_embedding = nn.Embedding(128, 32)
        self.lstm = nn.LSTM(32, 32)
        self.nonlinear = True

    def dimension(self):
        return 32

    def embed(self, q):
        b = torch.tensor([ord(c) for c in q] + [0] * (60 - len(q)),
                         dtype=torch.long).unsqueeze(1)
        e = self.char_embedding(b)
        _, (h, c) = self.lstm(e)
        return h.squeeze(2).squeeze(0)

class PreTrainedFeaturizer(nn.Module):
    def __init__(self, path):
        super().__init__()

        self.emb_model = LearnerValueFunction.load(path, map_location=torch.device('cpu'))
        self.emb_model.freeze()
        self.nonlinear = True

    def embed(self, q):
        return self.emb_model.embed_problems([q])[0]

    def dimension(self):
        return 128

class LinearModel(pl.LightningModule):
    def __init__(self, featurizer):
        super().__init__()
        self.featurizer = featurizer
        self.output = nn.Linear(featurizer.dimension(), 1)
        self.lr = self.learning_rate = 1e-5

    def forward(self, batch):
        embeddings = torch.cat([torch.tensor(self.featurizer.embed(q),
                                             device=self.device,
                                             dtype=torch.float).unsqueeze(0)
                                for q in batch])
        embeddings /= (embeddings**2).sum(dim=1).sqrt()[:, None]
        out = self.output(embeddings).squeeze(1)
        return out.sigmoid() if self.featurizer.nonlinear else out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y).sqrt()
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y).sqrt()
        self.log('val_loss', loss)
        return { 'val_loss': loss }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y).sqrt()
        self.log('test_loss', loss)
        return { 'test_loss': loss }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=(self.lr or self.learning_rate))

def run_experiment(config, log_wandb=True):
    if log_wandb:
        run = wandb.init(reinit=True, config=config)

    d = QuestionDifficultyDataset(config['dataset'], 10)
    train, val, test = split_train_val_test(d, config['train_portion'], 0,
                                            config.get('seed', 'test'))

    if config['featurizer'] == 'HandcraftedFeaturizer':
        f = HandcraftedFeaturizer()
    elif config['featurizer'] == 'PreTrainedFeaturizer':
        f = PreTrainedFeaturizer(config['pretrained_model_path'])
    elif config['featurizer'] == 'LSTMFeaturizer':
        f = LSTMFeaturizer()

    lm = LinearModel(f)

    trainer = pl.Trainer(logger=(WandbLogger('difficulty-prediction')
                                 if log_wandb
                                 else None),
#                         auto_lr_find=True,
                         max_epochs=20)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=128)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=128)

    trainer.tune(lm, train_dataloader)
    trainer.fit(lm, train_dataloader)
    results = trainer.test(lm, test_dataloader)
    print('Test results:', results)
    return results[0]['test_loss']

def run_repeated(config, k):
    results = []

    for i in range(k):
        results.append(run_experiment({ **config, 'seed': 'repeat-{}'.format(i) }, False))

    return np.mean(results), (1 - .95) * np.std(results)

def analyze_data_efficiency(config):
    data_points = collections.defaultdict(list)
    x = config['x']

    for x_i in x:
        model_config = { 'dataset': config['dataset'],
                         'train_portion': x_i,
                         'pretrained_model_path': config['pretrained_model_path'] }

        for model, featurizer in [('Handcrafted', 'HandcraftedFeaturizer'),
                                  ('Solver', 'PreTrainedFeaturizer'),
                                  ('LSTM', 'LSTMFeaturizer')]:
            print('Running', model, 'with trainining fraction', x_i)
            rmse = run_repeated({ **model_config, 'featurizer': featurizer }, config['n_repeats'])
            data_points[model].append((x_i, rmse))

    fig, ax = plt.subplots()
    ax.set_title('RMSE predicting % correct with different amounts of data')
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Fraction of data points used for training')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(data_points)))

    for (key, values), c in zip(data_points.items(), colors):
        # Mean
        ax.plot(x, [p[1][0] for p in values], color=c, label=key)
        ax.fill_between(x,
                        [p[1][0] - p[1][1] for p in values],
                        [p[1][0] + p[1][1] for p in values],
                        alpha=.1,
                        color=c)
    ax.legend()
    fig.savefig(config['output'])
    return fig

def analyze_representation_evolution(config):
    data_points = collections.defaultdict(list)
    x = config['x']
    paths = config['paths']

    for x_i, p_i in zip(x, paths):
        model_config = { 'dataset': config['dataset'],
                         'train_portion': 0.8,
                         'featurizer': 'PreTrainedFeaturizer',
                         'pretrained_model_path': p_i }

        rmse = run_repeated(model_config, config['n_repeats'])
        data_points['Solver'].append((x_i, rmse))

    fig, ax = plt.subplots()
    ax.set_title('RMSE predicting % correct on different iterations of the learned solver')
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Iteration')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(data_points)))

    for (key, values), c in zip(data_points.items(), colors):
        ax.plot(x, [p[1][0] for p in values], color=c, label=key)
        ax.fill_between(x,
                        [p[1][0] - p[1][1] for p in values],
                        [p[1][0] + p[1][1] for p in values],
                        alpha=.1,
                        color=c)
    fig.savefig(config['output'])
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--train', help='Train and evaluate one model',
                        action='store_true')
    parser.add_argument('--data-efficiency', help='Evaluate models on data efficiency.',
                        action='store_true')
    parser.add_argument('--representation-evolution', help='Evaluate many versions of our representation.',
                        action='store_true')

    opt = parser.parse_args()
    config = json.load(open(opt.config))

    if opt.train:
        train_and_eval(config)
    elif opt.data_efficiency:
        analyze_data_efficiency(config)
    elif opt.representation_evolution:
        analyze_representation_evolution(config)
