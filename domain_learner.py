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

# Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, base, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def tag_problem(s):
    return '<P> ' + s

def tag_step(s):
    return '<S> ' + s

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
        self.hidden_dim = hidden_dim = params.get('hidden_dim', 256)
        self.state_action_pairs = params.get('state_action_pairs', False)

        if self.kind == 'transformer':
            self.step_positional_encoding = PositionalEncoding(
                    self.embedding_dim,
                    params.get('positional_encoding_base', 10000))

            step_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=params.get('step_heads', 4),
                dim_feedforward=hidden_dim)

            self.step_encoder = nn.TransformerEncoder(step_encoder_layer,
                                                      num_layers=params.get('step_layers', 4))

            self.sol_positional_encoding = PositionalEncoding(
                    self.embedding_dim,
                    params.get('positional_encoding_base', 10000))

            sol_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=params.get('solution_heads', 4),
                dim_feedforward=hidden_dim)

            self.sol_encoder = nn.TransformerEncoder(sol_encoder_layer,
                                                     num_layers=params.get('sol_layers', 4))

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
        self.max_line_length = params.get('max_line_length', 100)

    def embed_states(self, x):
        embedding, lens = self.embed_batch(x)
        embedding = embedding.transpose(0, 1)
        embedding = self.step_positional_encoding(embedding * math.sqrt(self.embedding_dim))
        src_mask = self.generate_square_subsequent_mask(embedding.size(0)).to(self.device)
        step_encoder_out = self.step_encoder(embedding, src_mask)
        state_embeddings = step_encoder_out[torch.tensor(lens, device=embedding.device),
                                            torch.arange(embedding.shape[1], device=embedding.device),
                                            :]
        return state_embeddings

    def embed_steps(self, steps):
        return self.embed_states([self.abbreviate(tag_step(s)) for s in steps])

    def embed_problems(self, problems):
        return self.embed_states([self.abbreviate(tag_problem(p)) for p in problems])

    def embed_batch(self, batch):
        return self.encoding.embed_batch(batch, self.device)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def preprocess_example(self, x_i):
        if self.state_action_pairs:
            x_i = [tag_problem(x_i[-2]), tag_step(x_i[-1])]

        return [self.abbreviate(s) for s in x_i]

    def abbreviate(self, s):
        if len(s) > self.max_line_length:
            return s[:self.max_line_length] + '...'
        return s

    def forward(self, x):
        x = [self.preprocess_example(x_i) for x_i in x]

        if self.kind == 'transformer':
            # x is a list of lists of strings.
            batch_size = len(x)
            n_steps = [len(sol) for sol in x]
            max_steps = max(n_steps)
            x_cat = [step for sol in x for step in sol]

            embedding, lens = self.embed_batch(x_cat)
            embedding = embedding.transpose(0, 1)
            embedding = self.step_positional_encoding(embedding * math.sqrt(self.embedding_dim))
            src_mask = self.generate_square_subsequent_mask(embedding.size(0)).to(self.device)
            step_encoder_out = self.step_encoder(embedding, src_mask)
            step_encoder_out = step_encoder_out[torch.tensor(lens, device=embedding.device),
                                                torch.arange(embedding.shape[1], device=embedding.device),
                                                :]

            # Now step_encoder_out is of dimension sum(n_steps) x embedding_dim.
            m = []
            last_idx = 0

            for i in range(batch_size):
                m.append(
                        torch.cat([step_encoder_out[last_idx:last_idx + n_steps[i], :],
                                  torch.zeros((max_steps - n_steps[i] + 1, self.embedding_dim),
                                              device=embedding.device)]).unsqueeze(1)
                )
                last_idx += n_steps[i]

            step_embedding = torch.cat(m, dim=1)
            step_embedding = self.sol_positional_encoding(step_embedding)
            sol_src_mask = self.generate_square_subsequent_mask(step_embedding.size(0)).to(self.device)
            # step_embeddings.shape should be (max_steps + 1) x batch_size x embedding_dim.
            solution_encoder_out = self.sol_encoder(step_embedding, sol_src_mask)

            encoder_out = solution_encoder_out[torch.tensor(n_steps, device=embedding.device),
                                               torch.arange(step_embedding.shape[1], device=embedding.device),
                                               :]
        else:
            if isinstance(x[0], list):
                x = ['\n'.join(x_i) for x_i in x]

            embedding, lens = self.embed_batch(x)
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
            return torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.lr,
                                    betas=(0.9, 0.98), eps=1e-9)

def parse_solutions_dataset(path, max_example_size=0):
    print('Loading', path, '(max_example_size=', max_example_size, ')')
    with open(path) as f:
        d = json.load(f)

    examples = []
    solution_lens = []

    for row in d:
        if row['success']:
            solution_lens.append(len(row['solution']))

            for i in range(1, len(row['solution'])):
                examples.append((row['solution'][:i + 1][-max_example_size:], 1))

            for neg in row['negative-examples']:
                examples.append((neg[-max_example_size:], 0))

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

def collate_concat(l):
    x, y = zip(*l)
    return x, torch.tensor(y)

def train_domain_learner(config, gpus=0, logger=None):
    print('Training on', config['dataset'])
    _, examples, _ = parse_solutions_dataset(config['dataset'],
                                             config.get('max_example_size', 0))

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

    devices = GPUtil.getAvailable(order='memory', maxLoad=0.3, maxMemory=0.2)[:gpus]
    print('Using GPUs', devices)

    trainer = pl.Trainer(gpus=devices,
                         max_epochs=max_epochs,
                         logger=logger,
                         auto_lr_find=tune_lr)
    model = LearnerValueFunction(config['LearnerValueFunction'])

    print('Input:', 'state/action pairs' if model.state_action_pairs else 'full solution')

    if model.encoding.params.get('type') == 'bpe':
        print('Computing BPE vocabulary...')
        model.encoding.compute_vocabulary([x for x, y in examples])

    if tune_lr:
        print('Tuning learning rate')
        lr = (trainer.tuner.lr_find(model,
                                    DataLoader(train, batch_size=batch_size, collate_fn=collate_concat),
                                    DataLoader(val, batch_size=batch_size, collate_fn=collate_concat))
                           .suggestion())
        model.lr = lr
        print('Best learning rate found:', model.lr)

    trainer.fit(model,
                DataLoader(train, batch_size=batch_size, collate_fn=collate_concat),
                DataLoader(val, batch_size=batch_size, collate_fn=collate_concat))

    if config.get('output'):
        torch.save(model, config['output'])

def batch(l, batch_size):
    i = 0
    while i < len(l):
        yield l[i:i+batch_size]
        i += batch_size

def serve_model(config):
    gpus = GPUtil.getAvailable(order='memory', maxLoad=0.3, maxMemory=0.2)

    device = torch.device('cuda:{}'.format(gpus[0]) if len(gpus) else 'cpu')
    model = torch.load(config['model'], map_location=device)
    model.to(device)
    max_example_size = config.get('max_example_size', 0)
    port = config.get('port', 9911)

    print('Serving model on port', port, 'using device', device)
    print('Input:', 'state/action pairs' if model.state_action_pairs else 'full solution')

    log_requests_file = config.get('log_requests_to')
    if log_requests_file:
        print('Logging requests to', log_requests_file)
        f = open(log_requests_file, 'w')

    batch_size = config.get('batch_size', 64)
    app = Flask(__name__)

    @app.route('/', methods=['POST'])
    def serve():
        try:
            X = request.get_json()

            if log_requests_file:
                f.write('{}\n'.format(json.dumps(X)))
                f.flush()

            assert type(X) is list

            # If received a list of lists, possibly trim it first.
            X = [(x if isinstance(x, str) else x[-max_example_size:])
                 for x in X]

            y = []

            for b in batch(X, batch_size):
                assert len(b) <= batch_size
                y.extend(model(b).tolist())

            return json.dumps(y)
        except Exception as e:
            print('Server error:', e)
            import traceback; traceback.print_exc(e)
            with open('failed-requests', 'a') as f:
                f.write(json.dumps(X))
                f.write('\n')
            raise e

    app.run('127.0.0.1', config.get('port', port))

def now():
    return datetime.datetime.now().isoformat(timespec='seconds')

def learn_domain(config, gpus):
    wandb_run = wandb.init(config=config, project=f'domain-learner-{config["domain"]}')

    dataset = []
    stats = []
    step = config.get('depth_step', 1)
    server_port = config.get('server_port', 9911)
    server_address = 'http://127.0.0.1:{}/'.format(server_port)

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
                server_config['port'] = server_port
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
                    '-d', str(min(config['initial_depth'] + r*step,
                                  config.get('max_depth', 100))),
                    '-p', str(config['problems_per_round']),
                    '-S', server_address,
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

def compute_most_similar(embeddings):
    # L2-normalize.
    embeddings /= (embeddings**2).sum(axis=1).sqrt()[:, None]
    similarity = embeddings.matmul(embeddings.T)

    # Remove diagonal (self-similarity) and sort.
    similarity -= torch.eye(similarity.shape[0], device=similarity.device)
    return similarity.argsort(dim=1, descending=True).tolist()

def build_problem_graph(config, gpus):
    level_range = config.get('level_range', 3)
    max_problems_per_level = config.get('max_problems_per_level', 50)
    n_similar_problems = config.get('n_similar_problems', 10)
    n_similar_steps = config.get('n_similar_steps', 20)

    dataset, _, _ = parse_solutions_dataset(config['solutions_dataset'])

    n_solutions_by_level = collections.defaultdict(int)

    solutions_dataset = []

    random.shuffle(dataset)
    
    for s in dataset:
        if s['success']:
            level = (len(s['solution']) - 1) // level_range

            if n_solutions_by_level[level] < max_problems_per_level:
                solutions_dataset.append(s)
                n_solutions_by_level[level] += 1

    step_index_to_problem_step = {}

    for i, sol in enumerate(solutions_dataset):
        for j in range(1, len(sol['solution'])):
            step_index_to_problem_step[len(step_index_to_problem_step)] = (i, j)

    devices = GPUtil.getAvailable(order='memory', maxLoad=0.3, maxMemory=0.2)[:gpus]
    device = torch.device('cuda:{}'.format(devices[0]) if len(devices) else 'cpu')
    model = torch.load(config['model'], map_location=device)
    model.to(device)

    problems = [s['solution'][0] for s in solutions_dataset]
    n_steps = [len(s['solution']) - 1 for s in solutions_dataset]
    steps = [step for s in solutions_dataset
                  for step in s['solution'][1:]]

    print('Using', len(problems), 'problems with a total of', sum(n_steps), 'steps.')
    print('Computing problem similarity graph...')
    similar_problems = compute_most_similar(model.embed_problems(problems))

    for i, s in enumerate(solutions_dataset):
        s['similar_problems'] = similar_problems[i][:n_similar_problems]

    print('Computing step similarity graph...')
    similar_steps = compute_most_similar(model.embed_steps(steps))

    next_index = 0
    for sol in solutions_dataset:
        similar_steps_i = []
        for j in range(1, len(sol['solution'])):
            indices = [step_index_to_problem_step[idx]
                       for idx in similar_steps[next_index][:n_similar_steps]]
            for p, idx in indices:
                similar_steps_i.append({ 'problem': p, 'step': idx })
            next_index += 1
        sol['similar_steps'] = similar_steps_i

    with open(config['output'], 'w') as f:
        json.dump(solutions_dataset, f)

    print('Wrote', config['output'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Trains and serves the tutor domain learner')
    parser.add_argument('--train', action='store_const', default=False, const=True,
                        help='Train one round of the learner')
    parser.add_argument('--learn', action='store_const', default=False, const=True,
                        help='Learn a solver for the entire domain.')
    parser.add_argument('--build-graph', action='store_const', default=False, const=True,
                        help='Make the problem/solution/step graph that is used in the teaching game.')
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
    elif opt.build_graph:
        build_problem_graph(config, opt.gpus)
