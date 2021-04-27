'''LEGACY: this file is here for historical purposes, but isn't used anymore and will be
deleted at some point.'''

import argparse
import collections
import datetime
import json
import random
import os
import re
import math
import subprocess
import time
import numpy as np
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
from encoding import CharEncoding, PositionalEncoding


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
        self.state_delta_pairs = params.get('state_delta_pairs', False)

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
            self.gru_state = nn.GRU(self.embedding_dim,
                                    hidden_dim,
                                    params.get('layers', 2),
                                    batch_first=True,
                                    bidirectional=True)

            self.gru_action = nn.GRU(self.embedding_dim,
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

    def embed_state(self, state_batch):
        b, _ = self.embed_batch([self.abbreviate(s) for s in state_batch])
        gru_out, _ = self.gru_state(b)
        return gru_out[:, -1, :].reshape((len(state_batch), -1))

    def embed_action(self, action_batch):
        b, _ = self.embed_batch([self.abbreviate(s) for s in action_batch])
        gru_out, _ = self.gru_action(b)
        return gru_out[:, -1, :].reshape((len(action_batch), -1))

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
        elif self.state_delta_pairs:
            x_i = [tag_problem(x_i[-2]), tag_prob(x_i[-1])]

        return [self.abbreviate(s) for s in x_i]

    def abbreviate(self, s):
        if len(s) > self.max_line_length:
            return s[:self.max_line_length] + '...'
        return s

    def forward(self, state, action):
        if self.kind == 'transformer':
            # TODO adapt transformer to DRRN-like architecture.
            raise NotImplemented()

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
            state_embedding = self.embed_state(state)
            action_embedding = self.embed_action(action)
            return (state_embedding * action_embedding).sum(dim=1).sigmoid()

        return self.output(encoder_out).squeeze(1).sigmoid()

    def training_step(self, batch, batch_idx):
        state, action, y = batch
        y_hat = self(state, action)
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        state, action, y = batch
        y_hat = self(state, action).round()
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

    @staticmethod
    def load(*args, **kwargs):
        return torch.load(*args, **kwargs)

def parse_solutions_dataset(path, alpha=1.0):
    print('Loading', path)
    with open(path) as f:
        d = json.load(f)

    examples = []
    solution_lens = []

    for row in d:
        if row['success']:
            solution_lens.append(len(row['solution']))

            for i in range(1, len(row['solution'])):
                examples.append((row['solution'][i-1],
                                 row['solution-formal-description'][i],
                                 1 * alpha ** (len(row['solution']) - 1)))

            for neg in row['negative-examples']:
                examples.append((row['solution'][neg['index']],
                                 neg['step-formal-description'],
                                 0))

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
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])

def collate_concat(l):
    s, a, y = zip(*l)
    return s, a, torch.tensor(y)

def train_domain_learner(config, gpus=0, logger=None):
    print('Training on', config['dataset'])
    _, examples, _ = parse_solutions_dataset(config['dataset'],
                                             config.get('alpha', 1))

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

    devices = GPUtil.getAvailable(order='memory', maxLoad=0.3, maxMemory=0.5)[:gpus]
    print('Using GPUs', devices)

    trainer = pl.Trainer(gpus=devices if gpus else None,
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
    gpus = GPUtil.getAvailable(order='memory', maxLoad=0.3, maxMemory=0.5)

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
            X = [(x['state'], x['action']) for x in X]

            y = []

            for b in batch(X, batch_size):
                b_s, b_a = zip(*b)
                y.extend(model(b_s, b_a).tolist())

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
    domain = config["domain"]
    initial_policy = config["initial_policy"]

    print('Learning to solve', domain, 'starting with', initial_policy, 'policy')

    wandb_run = wandb.init(config=config, project=f'domain-learner-{domain}-{initial_policy}')

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
                    '-D', domain,
                    '-o', solver_output,
                    '-d', str(min(config['initial_depth'] + r*step,
                                  config.get('max_depth', 100))),
                    '-p', str(config['problems_per_round']),
                    '-S', server_address,
                    '-T', str(config.get('solver_threads', 8)),
                    '-b', str(config['beam_width']),
                    '-P', # <-- Policy, specified below depending on the case.
                    ]
            if use_value_function:
                # Use learned policy after bootstrap round.
                args.append('neural')
            else:
                args.append(initial_policy)

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

def compute_pairwise_similarities(embeddings):
    # L2-normalize.
    embeddings = embeddings / (embeddings**2).sum(axis=1).sqrt()[:, None]

    B = 1024
    N, E = embeddings.shape
    m = np.zeros((N, N))
    n_batches = (N + B - 1) // B

    print('Computing all pairwise similarities between', N, 'items.')
    for i in tqdm(range(n_batches)):
        i_b, i_e = i*B, min(N, (i+1)*B)
        m_i = embeddings[i_b:i_e, :]

        for j in range(i, n_batches):
            j_b, j_e = j*B, min(N, (j+1)*B)
            m_j = embeddings[j_b:j_e, :]

            similarities = m_i.matmul(m_j.T).cpu().numpy()
            m[i_b:i_e, j_b:j_e] = similarities
            m[j_b:j_e, i_b:i_e] = similarities.T

    return m

@torch.no_grad()
def build_problem_graph(config, gpus):
    level_range = config.get('level_range', 3)
    max_problems_per_level = config.get('max_problems_per_level', 50)
    n_similar_problems = config.get('n_similar_problems', 10)
    n_similar_steps = config.get('n_similar_steps', 20)
    n_distractors = config.get('n_distractors', 10)

    dataset, _, _ = parse_solutions_dataset(config['solutions_dataset'])
    dataset = [r for r in dataset if r['success']]

    random.shuffle(dataset)

    all_steps = []
    guess_step_exercises = []

    for p, s in enumerate(dataset):
        for i in range(len(s['solution']) - 1):
            s_id = '{}_{}'.format(p, i),
            all_steps.append({
                'problem': p,
                'id': s_id,
                'index': len(all_steps),
                'state': s['solution'][i],
                'step': s['solution'][i+1],
                'step-description': s['solution-description'][i+1],
                'q-value': s['solution-value'][i+1],
                'label': 'pos',
                'level': len(s['solution']) - (i + 1),
            })

            all_steps.append({
                'problem': p,
                'id': s_id,
                'index': len(all_steps),
                'state': s['solution'][i],
                'step': s['negative-examples'][i]['step'],
                'step-description': s['negative-examples'][i]['step-description'],
                'q-value': s['negative-examples'][i]['value'],
                'label': 'neg',
            })

            guess_step_exercises.append({
                'problem': p,
                'step': i,
                'state': s['solution'][i],
                'state-tex': s['solution-tex'][i],
                'pos': all_steps[-2],
                'neg': all_steps[-1],
            })

    print('Loading model...')
    devices = GPUtil.getAvailable(order='memory', maxLoad=0.3, maxMemory=0.5)[:gpus]
    device = torch.device('cuda:{}'.format(devices[0]) if len(devices) else 'cpu')
    model = torch.load(config['model'], map_location=device)
    model.to(device)

    step_texts = [s['step'] for s in all_steps]
    print(len(step_texts), 'steps.')
    step_embeddings = []
    for b in batched(step_texts):
        step_embeddings.append(model.embed_states(b))
    step_embeddings = torch.cat(step_embeddings)

    step_sim = compute_pairwise_similarities(step_embeddings)

    def get_variables(s):
        return ''.join(sorted(set(c for c in s if c.isalpha())))

    def rewrite_variable(s, var):
        return re.sub('[a-z]', var, s)

    for p in dataset:
        p['variables'] = get_variables(p['solution'][0])

    print('Finding distractors for guess_step_exercises...')
    for e in tqdm(guess_step_exercises):
        p_var = get_variables(e['state'])
        similar = np.flip(step_sim[e['pos']['index'], :].argsort())
        valid = [all_steps[s]
                 for s in similar
                 if (dataset[e['problem']]['variables'] ==
                     dataset[all_steps[s]['problem']]['variables'])]
        e['distractors'] = [{ **d, 'step': rewrite_variable(d['step'], p_var)}
                            for d in valid[:n_distractors]
                            if d['step-description'] not in
                              (e['pos']['step-description'],
                               e['neg']['step-description'])]
        e['pos_neg_sim'] = step_sim[e['pos']['index'], e['neg']['index']]

    guess_step_exercises = [e for e in guess_step_exercises
                            if len(e['distractors']) > 0]

    guess_state_exercises = []

    print('Creating guess_state_exercises...')
    for i, e in enumerate(tqdm(guess_step_exercises)):
        ex = {
            'step-description': e['pos']['step-description'],
            'correct': e['state'],
        }

        p_var = get_variables(e['state'])
        similar = np.flip(step_sim[e['pos']['index'], :].argsort())

        valid = [all_steps[s]
                 for s in similar
                 if (dataset[e['problem']]['variables'] ==
                     dataset[all_steps[s]['problem']]['variables'])]

        ex['distractors'] = [rewrite_variable(d['step'], p_var)
                             for d in valid[:n_distractors]]
        ex['distractors'] = [d for d in ex['distractors'] if d != ex['correct']]

        if len(ex['distractors']) > 0:
            guess_state_exercises.append(ex)

    print(len(guess_step_exercises), '"guess the step" exercises')
    print(len(guess_state_exercises), '"guess the state" exercises')

    with open(config['output'], 'w') as f:
        json.dump({ 'guess_step_exercises': guess_step_exercises,
                    'guess_state_exercises': guess_state_exercises }, f)

    print('Wrote', config['output'])

def sample_exercises(config):
    with open(config['dataset']) as f:
        all_exercises = json.load(f)

    exercises = {
        k: random.sample(v, config['n'])
        for k, v in all_exercises.items()
    }

    with open(config['output'], 'w') as f:
        json.dump(exercises, f)

def batched(seq, b=128):
    i = 0
    while i < len(seq):
        yield seq[i:i+b]
        i += b

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
    parser.add_argument('--sample', help='Sample exercises.', action='store_const',
                        default=False, const=True)
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
    elif opt.sample:
        sample_exercises(config)
