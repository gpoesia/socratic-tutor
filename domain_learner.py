#!/usr/bin/python

import argparse
import collections
import datetime
import json
import random
import os
import math
import subprocess
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import GPUtil
from pytorch_lightning.loggers import WandbLogger
from flask import Flask, request
import wandb
from tqdm import tqdm

MAX_EXAMPLE_SIZE = 5

class CharEncoding(nn.Module):
    def __init__(self, params={}):
        super().__init__()

        self.params = params
        self.embedding = nn.Embedding(128, params.get('embedding_dim', 64))

        self.padding_idx = 0
        self.end_token_idx = 1

    def embed_batch(self, batch, device=None):
        lens = [len(s) for s in batch]
        max_len = max(lens)
        int_batch = torch.LongTensor(
            [list(s.encode('ascii')) + [self.end_token_idx] + [self.padding_idx] * (max_len - len(s))
             for s in batch])
        return self.embedding(int_batch.to(device=device)), lens

class BytePairEncoding(nn.Module):
    NUM_ASCII = 128
    PADDING_INDEX = 0
    START_INDEX = 1
    END_INDEX = 2

    def __init__(self, params={}):
        super().__init__()
        self.params = params
        self.v_size = params.get('vocabulary_size', 1000)
        self.e_size = params.get('embedding_size', 256)
        self.vocab = {}
        self.piece2ind = {}
        self.embedding = nn.Embedding(
            self.v_size, self.e_size, padding_idx=self.PADDING_INDEX)

        if params.get('vocab_file'):
            self.load_vocabulary(params['vocab_file'])
        self.keep_digits = params.get('keep_digits', False)
        self.max_examples = params.get('max_examples', 1000)

    def compute_vocabulary(self, examples):
        frequencies = collections.defaultdict(int)

        for i in range(self.NUM_ASCII):
            self.vocab[i] = chr(i)
            self.piece2ind[chr(i)] = i

        if self.max_examples:
            examples = random.sample(examples, k=min(len(examples), self.max_examples))

        ds = [list(e) for e in examples]

        print('Computing Byte Pair Encoding vocabulary...')
        for _ in tqdm(range(self.v_size - len(self.vocab))):
            freq = collections.defaultdict(int)

            for l in ds:
                for i in range(len(l) - 1):
                    if not ((l[i].isdigit() or l[i+1].isdigit()) and self.keep_digits):
                        freq[(l[i], l[i+1])] += 1

            pair, _ = max(list(freq.items()), key=lambda p: p[1])
            piece = pair[0] + pair[1]
            index = len(self.vocab)
            self.vocab[index] = piece
            self.piece2ind[piece] = index

            for i in range(len(ds)):
                j, new_l = 0, []
                s = ds[i]

                while j < len(s):
                    if j + 1 < len(s) and s[j] == pair[0] and s[j+1] == pair[1]:
                        new_l.append(piece)
                        j += 2
                    else:
                        new_l.append(s[j])
                        j += 1

                ds[i] = new_l

    def encode_indices(self, s, device=None):
        'Given a string, returns a 2D tensor representation of it.'
        i, l = 0, []

        while i < len(s):
            last = self.piece2ind[s[i]]

            for j in range(i+1, len(s) + 1):
                index = self.piece2ind.get(s[i:j])
                if index is None:
                    break
                last = index

            l.append(last)
            i = j

        l_t = torch.tensor([self.START_INDEX] +
                           l +
                           [self.END_INDEX],
                            dtype=torch.long, device=device)
        return l_t

    def encode(self, s):
        idxs = self.encode_indices(s)
        return self.embedding(idxs)

    def embed(self, indices):
        return self.embedding(indices)

    def padding_token_index(self):
        return self.PADDING_INDEX

    def embed_batch(self, batch, device=None):
        bi, lens = self.encode_batch_indices(batch, device)
        return self.embed(bi), lens

    def encode_batch_indices(self, batch, device=None):
        if len(batch) == 0:
            return torch.zeros((0, 0), device=device)

        indices = [self.encode_indices(s, device) for s in batch]
        lens = [len(t) - 1 for t in indices]

        max_length = max(map(len, indices))
        padding_tensor = torch.tensor([self.PADDING_INDEX],
                                      dtype=torch.long, device=device)
        bi = torch.stack(
                [torch.cat([idx, padding_tensor.repeat(max_length - len(idx))])
                 for idx in indices])

        return bi, lens

    def dump_vocabulary(self, path):
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'piece2ind': self.piece2ind
            }, f)

    def load_vocabulary(self, path):
        with open(path) as f:
            v = json.load(f)
            self.vocab = v['vocab']
            self.piece2ind = v['piece2ind']

# Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnerValueFunction(pl.LightningModule):
    def __init__(self, params={}):
        super().__init__()

        self.params = params
        self.embedding_dim = params.get('embedding_dim', 64)

        encoding_params = params.get('encoding', {})
        if encoding_params.get('type', 'char') == 'char':
            self.encoding = CharEncoding({ **encoding_params,
                                           'embedding_dim': self.embedding_dim })
        else:
            self.encoding = BytePairEncoding({ **encoding_params,
                                               'embedding_dim': self.embedding_dim })

        self.kind = params.get('kind', 'transformer')
        hidden_dim = params.get('hidden_dim', 256)

        if self.kind == 'transformer':
            self.positional_encoding = PositionalEncoding(self.embedding_dim)

            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=params.get('heads', 4),
                dim_feedforward=hidden_dim)

            self.encoder = nn.TransformerEncoder(self.encoder_layer,
                                                 num_layers=params.get('layers', 4))
            out_dim = self.embedding_dim
        else:
            self.gru = nn.GRU(self.embedding_dim,
                              hidden_dim,
                              params.get('layers', 2),
                              batch_first=True,
                              bidirectional=True)
            out_dim = 2*hidden_dim

        self.output = nn.Linear(out_dim, 1)

        self.lr = params.get('lr', 1e-3)

    def embed_batch(self, batch):
        return self.encoding.embed_batch(batch, self.device)

    def forward(self, x):
        embedding, lens = self.embed_batch(x)

        if self.kind == 'transformer':
            embedding = self.positional_encoding(embedding)
            encoder_out = self.encoder(embedding)
            s_len = embedding.shape[1]
            encoder_out = encoder_out[torch.arange(embedding.shape[0],
                                                   device=embedding.device), # batch size
                                      torch.tensor(lens,
                                                   device=embedding.device), # lengths
                                      :]
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
        optimizer = self.params.get('optimizer', 'Adam')
        if optimizer == 'SGD':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            return torch.optim.SGD(self.parameters(), lr=self.lr)

def parse_solutions_dataset(path, verbose=False):
    with open(path) as f:
        d = json.load(f)

    examples = []
    solution_lens = []

    for row in d:
        if row['success']:
            solution_lens.append(len(row['solution']))

            for i in range(len(row['solution'])):
                examples.append(('\n'.join(row['solution'][:i + 1][-MAX_EXAMPLE_SIZE:]), 1))

            for neg in row['negative-examples']:
                examples.append(('\n'.join(neg[-MAX_EXAMPLE_SIZE:]), 0))

    max_solution_len = max(solution_lens)
    len_hist = collections.Counter(solution_lens)

    return (d,
            examples,
            {
                'n': len(d),
                'avg_solution_len': sum(solution_lens) / len(solution_lens),
                'max_solution_len': max_solution_len,
                'success_rate': len(solution_lens) / len(d),
                'solution_len_hist': [len_hist.get(l, 0) for l in range(0, max_solution_len + 1)]
            })

def split_dataset(dataset):
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

def train_domain_learner(config, gpus=0, logger=None):
    print('Training on', config['dataset'])
    _, examples, _ = parse_solutions_dataset(config['dataset'])

    if config.get('max_examples'):
        print('Limiting number of examples to', config['max_examples'])
        examples = examples[:config['max_examples']]

    train, val = split_dataset(examples)
    batch_size = config.get('batch_size', 128)
    max_epochs = config.get('max_epochs', 50)
    tune_lr = config.get('tune_lr', False)

    if logger is None:
        logger = WandbLogger()

    logger.log_hyperparams(config)

    devices = GPUtil.getAvailable(order='random', maxLoad=0.3, maxMemory=0.1)[:gpus]
    print('Using GPUs', devices)

    trainer = pl.Trainer(gpus=devices,
                         max_epochs=max_epochs,
                         logger=logger,
                         auto_lr_find=tune_lr)
    model = LearnerValueFunction(config['LearnerValueFunction'])

    if model.encoding.params.get('type') == 'bpe':
        print('Computing BPE vocabulary...')
        model.encoding.compute_vocabulary([x for x, y in examples])

    if tune_lr:
        print('Tuning learning rate')
        lr = (trainer.tuner.lr_find(model,
                                    DataLoader(train, batch_size=batch_size),
                                    DataLoader(val, batch_size=batch_size))
                           .suggestion())
        model.lr = lr
        print('Best learning rate found:', model.lr)

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
    gpus = GPUtil.getAvailable(order='random', maxLoad=0.3, maxMemory=0.2)

    device = torch.device('cuda:{}'.format(gpus[0]) if len(gpus) else 'cpu')
    model = torch.load(config['model'], map_location=device)
    model.to(device)

    print('Serving model on', device)

    batch_size = config.get('batch_size', 64)
    app = Flask(__name__)

    @app.route('/', methods=['POST'])
    def serve():
        X = request.get_json()

        assert type(X) is list

        # If received a list of lists, join it first.
        X = [(x if isinstance(x, str) else '\n'.join(x[-MAX_EXAMPLE_SIZE:]))
             for x in X]

        y = []

        for b in batch(X, batch_size):
            y.extend(model(b).tolist())

        return json.dumps(y)

    app.run('127.0.0.1', config.get('port', 9911))

def now():
    return datetime.datetime.now().isoformat(timespec='seconds')

def learn_domain(config, gpus):
    wandb_run = wandb.init(config=config, project=f'domain-learner-{config["domain"]}')

    dataset = []
    stats = []

    for r in range(config['rounds']):
        print(now(), '#' * 20, 'Round', r+1, '/', config['rounds'])

        # Run solver for this round.
        solver_output = config['solver_output'].format(r)
        if os.path.exists(solver_output):
            print(solver_output, 'already exists. Skipping.')
        else:
            # We use the learned value function starting from the second round.
            use_value_function = r > 0

            # If we need the value function to run the solver, spawn a server.
            if use_value_function:
                server_config = config['server_template']
                server_config['model'] = config['learner_template']['output'].format(r-1)
                server_config_path = '{}-server-{}.json'.format(config['domain'], r)
                with open(server_config_path, 'w') as f:
                    json.dump(server_config, f)

                print(now(), 'Spawning value function server and giving 60s for it to come up...')
                server_process = subprocess.Popen(['python', 'domain_learner.py',
                                                   '--serve',
                                                   '--config', server_config_path],
                                                   stderr=subprocess.DEVNULL)
                time.sleep(60)

            print(now(), 'Running solver...')
            args = ['racket', 'run-learn.rkt',
                    '-o', solver_output,
                    '-d', str(config['initial_depth'] + r),
                    # Use negative examples = depth.
                    '-n', str(config['initial_depth'] + r),
                    '-p', str(config['problems_per_round']),
                    '-b', str(config['beam_width'])]
            if use_value_function:
                # Use value function after bootstrap round.
                args.append('-V')
            print(now(), '$', ' '.join(args))
            subprocess.run(args)

            # Kill the model server.
            if use_value_function:
                print(now(), 'Terminating value function server.')
                server_process.terminate()

        # Merge solver output dataset with datasets from previous rounds.
        round_dataset, _, round_stats = parse_solutions_dataset(solver_output)

        print(now(), 'Solver statistics for round', r, ':\n', json.dumps(round_stats, indent=4))
        stats.append(round_stats)
        dataset.extend(round_dataset)

        dataset_path = config['learner_template']['dataset'].format(r)
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f)

        print(now(), 'Dataset now has', sum(row['success'] for row in dataset), 'solutions.')

        learner_config = config['learner_template'].copy()
        model_output = learner_config['output'].format(r)

        if os.path.exists(model_output):
            print('Model', model_output, 'already exists. Skipping.')
        else:
            learner_config['dataset'] = dataset_path
            learner_config['output'] = model_output
            train_domain_learner(learner_config, gpus)

        wandb_run.log({ 'avg_solution_len': round_stats['avg_solution_len'],
                        'success_rate': round_stats['success_rate'] }, step=r+1)

    stats_path = '{}-stats.json'.format(config['domain'])

    with open(stats_path, 'w') as f:
        json.dump(stats, f)

    print(now(), 'Wrote', stats_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Trains and serves the tutor domain learner')
    parser.add_argument('--train', action='store_const', default=False, const=True,
                        help='Train one round of the learner')
    parser.add_argument('--learn', action='store_const', default=False, const=True,
                        help='Learn a solver for the entire domain.')
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
    elif opt.learn:
        learn_domain(config, opt.gpus)
