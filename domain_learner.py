#!/usr/bin/python

import argparse
import json
import random
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import GPUtil
from pytorch_lightning.loggers import WandbLogger
from flask import Flask, request


class CharEncoding(nn.Module):
    def __init__(self, params={}):
        super().__init__()

        self.embedding = nn.Embedding(128, params.get('embedding_dim', 64))

        self.padding_idx = 0
        self.end_token_idx = 1

    def embed_batch(self, batch, device=None):
        max_len = max(len(s) for s in batch)
        int_batch = torch.LongTensor(
            [list(s.encode('ascii')) + [self.end_token_idx] + [self.padding_idx] * (max_len - len(s))
             for s in batch])
        return self.embedding(int_batch.to(device=device))

class LearnerValueFunction(pl.LightningModule):
    def __init__(self, params={}):
        super().__init__()

        self.params = params
        self.encoding = CharEncoding(params.get('char_encoding', {}))
        embedding_dim = params.get('embedding_dim', 64)

        self.kind = params.get('kind', 'transformer')
        hidden_dim = params.get('hidden_dim', 256)

        if self.kind == 'transformer':
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=params.get('heads', 4),
                dim_feedforward=hidden_dim)

            self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                                 num_layers=params.get('layers', 4))
            out_dim = embedding_dim
        else:
            self.gru = nn.GRU(embedding_dim,
                              hidden_dim,
                              params.get('layers', 2),
                              batch_first=True,
                              bidirectional=True)
            out_dim = 2*hidden_dim

        self.output = nn.Linear(out_dim, 1)

    def embed_batch(self, batch):
        return self.encoding.embed_batch(batch, self.device)

    def forward(self, x):
        embedding = self.embed_batch(x)

        if self.kind == 'transformer':
            encoder_out = self.encoder(embedding)
            s_len = embedding.shape[1]
            encoder_out = encoder_out[:, 0, :]
        else:
            gru_out, _ = self.gru(embedding)
            encoder_out = gru_out[:, -1, :]

        return self.output(encoder_out).squeeze(1).sigmoid()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).round()
        acc = (y == y_hat).float().mean()
        self.log('val_acc', acc)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def parse_solutions_dataset(path, verbose=False):
    with open(path) as f:
        d = json.load(f)

    examples = []
    solution_lens = []

    for row in d:
        if row['success']:
            solution_lens.append(len(row['solution']))
            
            for i in range(len(row['solution'])):
                examples.append(('\n'.join(row['solution'][:i + 1]), 1))

            for neg in row['negative-examples']:
                examples.append(('\n'.join(neg), 0))

    print('Average solution length:', sum(solution_lens) / len(solution_lens))

    return examples

def split_dataset(dataset):
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

def train_domain_learner(config, gpus=0, logger=None):
    print('Training on', config['dataset'])
    examples = parse_solutions_dataset(config['dataset'])
    train, val = split_dataset(examples)
    batch_size = config.get('batch_size', 128)

    if logger is None:
        logger = WandbLogger()

    trainer = pl.Trainer(gpus=GPUtil.getAvailable(order='random', maxLoad=0.3, maxMemory=0.5)[:gpus],
                         logger=logger)
    model = LearnerValueFunction(config['LearnerValueFunction'])

    trainer.fit(model,
                DataLoader(train, batch_size=batch_size),
                DataLoader(val, batch_size=batch_size))

    if config.get('output'):
        torch.save(model, config['output'])

def batch(l, batch_size):
    i = 0
    while i < len(l):
        yield l[i:i+batch_size]
        i += batch_size

def serve_model(config):
    gpus = GPUtil.getAvailable(order='random', maxLoad=0.3, maxMemory=0.5)

    device = torch.device('cuda:{}'.format(gpus[0]) if len(gpus) else 'cpu')
    model = torch.load(config['model'], map_location=device)
    model.to(device)

    print('Serving model on', device)

    batch_size = config.get('batch_size', 64)
    app = Flask(__name__)

    @app.route('/', methods=['POST'])
    def serve():
        X = request.get_json()

        print('Request:', X)

        assert type(X) is list

        # If received a list of lists, join it first.
        X = [(x if isinstance(x, str) else '\n'.join(x))
             for x in X]

        y = []

        for b in batch(X, batch_size):
            y.extend(model(b).tolist())

        return json.dumps(y)

    app.run('127.0.0.1', config.get('port', 9911))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Trains and serves the tutor domain learner')
    parser.add_argument('--train', action='store_const', default=False, const=True,
                        help='Train one round of the learner')
    parser.add_argument('--serve', action='store_const', default=False, const=True,
                        help='Serve a ranking model')
    parser.add_argument('--dataset', help='Solutions dataset to use.')
    parser.add_argument('--gpus', help='Number of GPUs to use', type=int, default=0)
    parser.add_argument('--config', help='Path to config file.')

    opt = parser.parse_args()

    if opt.config:
        with open(opt.config) as f:
            config = json.load(f)
    else:
        config = {}

    if opt.train:
        train_domain_learner(config, opt.gpus)
    elif opt.serve:
        serve_model(config)
