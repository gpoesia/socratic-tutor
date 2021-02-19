import json
import argparse
import collections
import sys
import random

# sys.path.append('variational-item-response-theory-public')

import os
import time
import math
import numpy as np
from tqdm import tqdm
import numpy as np
import torch
from torch import optim
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification \
    import auroc as pl_auroc, accuracy as pl_accuracy
import GPUtil

# import pyro
# from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
# from pyro.infer import Importance, EmpiricalMarginal
# from pyro.optim import Adam
import wandb

#from src.pyro_core.models import (
#    VIBO_3PL,
#    VIBO_2PL,
#    VIBO_1PL,
#)
#from src.datasets import load_dataset
#from src.utils import AverageMeter, save_checkpoint
#from src.config import OUT_DIR

import dataset
from domain_learner \
    import LearnerValueFunction, CharEncoding, PositionalEncoding, collate_concat, batched

### Models ###

def cos_similarity(v1, v2):
    return (v1 * v2).sum() / (v1**2).sum().sqrt() / (v2**2).sum().sqrt()

class EKT(pl.LightningModule):
    def __init__(self, config, n_questions):
        super().__init__()

        hidden_dim = config.get('hidden_size', 128)

        self.q_embed_matrix = nn.Embedding(n_questions, hidden_dim)
        self.k = config.get('k', 3)

        self.lstm = nn.LSTMCell(2*hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_questions)
        self.lr = config.get('lr', 1e-3)

    def predict_student(self, questions, responses):
        preds = []

        for i, (q, r) in enumerate(zip(questions, responses)):
            p, sd, sc = torch.tensor(0.0), 1e-6, 0
            e_i = self.q_embed_matrix.weight[q, :]

            neighbors = [(-np.inf, 0.5)]

            for pq in range(i):
                e_pq = self.q_embed_matrix.weight[pq, :]
                d = cos_similarity(e_i, e_pq)
                neighbors.append((d, responses[pq]))

            neighbors.sort(key=lambda it: -it[0])
            preds.append(np.mean([float(r) for d, r in neighbors[:self.k]]))

        return preds

    def forward(self, q_data, qa_data):
        B, L = q_data.shape
        preds = []

        for j in range(B):
            preds.append(self.predict_student(q_data[j, :], qa_data[j, :]))

        return torch.tensor(preds)

    def get_loss(self, preds, qs, response, mask):
        pred_q = preds.reshape(-1)
        response = response[:, 1:].reshape(-1)
        mask = mask[:, 1:].reshape(-1)
        index = torch.where(mask)[0]

        pred_q = torch.gather(pred_q, 0, index)
        response = torch.gather(response, 0, index)

        loss = F.binary_cross_entropy_with_logits(pred_q, response.float())
        acc = pl_accuracy(pred_q.sigmoid().round(), response)
        auroc = pl_auroc(pred_q.sigmoid(), response)

        return loss, acc, auroc

    def training_step(self, batch, batch_idx):
        index, response, problem_id, mask = batch
        response = response.to(self.device)
        problem_id = problem_id.to(self.device)
        mask = mask.to(self.device)
        preds = self(problem_id, response)

        loss, accuracy, auroc = self.get_loss(preds, problem_id, response, mask)

        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        self.log('train_auroc', auroc)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx, prefix='test'):
        index, response, problem_id, mask = batch
        response = response.to(self.device)
        problem_id = problem_id.to(self.device)
        mask = mask.to(self.device)
        preds = self(problem_id, response)

        _, accuracy, auroc = self.get_loss(preds, problem_id, response, mask)
        metrics = { f'{prefix}_accuracy': accuracy, f'{prefix}_auroc': auroc }
        self.log_dict(metrics)

        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class DKT(pl.LightningModule):
    def __init__(self, config, n_questions):
        super().__init__()

        hidden_dim = config.get('hidden_size', 128)

        self.q_embed_matrix = nn.Embedding(n_questions, hidden_dim)
        self.a_embedding = nn.Embedding(2, hidden_dim)

        self.lstm = nn.LSTMCell(2*hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_questions)
        self.lr = config.get('lr', 1e-3)

    def forward(self, q_data, qa_data):
        B, L = q_data.shape

        pred = []

        q_emb = self.q_embed_matrix(q_data)
        a_emb = self.a_embedding(qa_data.relu())

        hc = None

        for i in range(L):
            X_i = torch.cat([q_emb[:, i, :], a_emb[:, i, :]], dim=1)
            hc = (h, c) = self.lstm(X_i, hc)
            pred.append(self.output(h).unsqueeze(1))

        return torch.cat(pred, dim=1)

    def get_loss(self, preds, qs, response, mask):
        pred_q = torch.gather(preds, 2, qs.unsqueeze(2)).squeeze(2)[:, :-1].reshape(-1)
        response = response[:, 1:].reshape(-1)
        mask = mask[:, 1:].reshape(-1)
        index = torch.where(mask)[0]

        pred_q = torch.gather(pred_q, 0, index)
        response = torch.gather(response, 0, index)

        loss = F.binary_cross_entropy_with_logits(pred_q, response.float())
        acc = pl_accuracy(pred_q.sigmoid().round(), response)
        auroc = pl_auroc(pred_q.sigmoid(), response)

        return loss, acc, auroc

    def training_step(self, batch, batch_idx):
        index, response, problem_id, mask = batch
        response = response.to(self.device)
        problem_id = problem_id.to(self.device)
        mask = mask.to(self.device)
        preds = self(problem_id, response)

        loss, accuracy, auroc = self.get_loss(preds, problem_id, response, mask)

        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        self.log('train_auroc', auroc)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx, prefix='test'):
        index, response, problem_id, mask = batch
        response = response.to(self.device)
        problem_id = problem_id.to(self.device)
        mask = mask.to(self.device)
        preds = self(problem_id, response)

        _, accuracy, auroc = self.get_loss(preds, problem_id, response, mask)
        metrics = { f'{prefix}_accuracy': accuracy, f'{prefix}_auroc': auroc }
        self.log_dict(metrics)

        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# Adapted from Mike's code in VIBO repository:
#
# https://github.com/mhw32/variational-item-response-theory-public

class DKVMN_IRT(pl.LightningModule):
    """Adapted from the TensorFlow implementation.
    https://github.com/ckyeungac/DeepIRT/blob/master/model.py
    """

    def __init__(
            self,
            config,
            device,
            batch_size,
            n_questions,
            memory_size,
        ):
        super().__init__()

        self.max_batch_size = batch_size
        self.n_questions = n_questions
        self.memory_size = memory_size
        self.memory_key_state_dim = config['hidden_size']
        self.memory_value_state_dim = config['hidden_size']
        self.summary_vector_output_dim = config['hidden_size']

        self.init_key_memory = torch.randn(self.memory_size, self.memory_key_state_dim).to(device)
        self.init_value_memory = torch.randn(self.memory_size, self.memory_value_state_dim).to(device)

        self.memory = DKVMN(
            self.memory_size,
            self.memory_key_state_dim,
            self.memory_value_state_dim,
            self.init_key_memory,
            self.init_value_memory.unsqueeze(0).repeat(batch_size, 1, 1),
        )
        self.q_embed_matrix = nn.Embedding(
            self.n_questions,
            self.memory_key_state_dim,
        )
        self.qa_embed_matrix = nn.Embedding(
            2, # * self.n_questions + 1,
            self.memory_value_state_dim,
        )
        self.summary_vector_fc = nn.Linear(
            self.memory_key_state_dim + self.memory_value_state_dim,
            self.summary_vector_output_dim,
        )
        self.student_ability_fc = nn.Linear(
            self.summary_vector_output_dim,
            1,
        )
        self.question_difficulty_fc = nn.Linear(
            self.memory_key_state_dim,
            1,
        )
        self.lr = config.get('lr', 0.001)

    def forward(self, q_data, qa_data):
        """
        q_data  : (batch_size, seq_len)
        qa_data : (batch_size, seq_len)
        label   : (batch_size, seq_len)
        """
        batch_size, seq_len = q_data.size(0), q_data.size(1)

        if batch_size < self.max_batch_size:
            q_data = torch.cat([q_data,
                                torch.zeros((self.max_batch_size - batch_size,
                                             q_data.shape[1]),
                                             device=q_data.device)])
            qa_data = torch.cat([qa_data,
                                 torch.zeros((self.max_batch_size - batch_size,
                                              qa_data.shape[1]),
                                              device=q_data.device)])

        q_embed_data  = self.q_embed_matrix(q_data.long())
        qa_embed_data = self.qa_embed_matrix((0*q_data + qa_data.relu()).long())

        sliced_q_embed_data = torch.chunk(q_embed_data, seq_len, dim=1)
        sliced_qa_embed_data = torch.chunk(qa_embed_data, seq_len, dim=1)

        pred_zs, student_abilities, question_difficulties = [], [], []

        for i in range(seq_len):
            q = sliced_q_embed_data[i].squeeze(1)
            qa = sliced_qa_embed_data[i].squeeze(1)

            correlation_weight = self.memory.attention(q)
            read_content = self.memory.read(correlation_weight)
            new_memory_value = self.memory.write(correlation_weight, qa)

            mastery_level_prior_difficulty = torch.cat([read_content, q], dim=1)

            summary_vector = self.summary_vector_fc(
                mastery_level_prior_difficulty,
            )
            summary_vector = torch.tanh(summary_vector)
            student_ability = self.student_ability_fc(summary_vector)
            question_difficulty = self.question_difficulty_fc(q)
            question_difficulty = torch.tanh(question_difficulty)

            pred_z = 3.0 * student_ability - question_difficulty

            pred_zs.append(pred_z)
            student_abilities.append(student_ability)
            question_difficulties.append(question_difficulty)

        pred_zs = torch.cat(pred_zs, dim=1)
        student_abilities = torch.cat(student_abilities, dim=1)
        question_difficulties = torch.cat(question_difficulties, dim=1)

        return (pred_zs[:batch_size],
                student_abilities[:batch_size],
                question_difficulties[:batch_size])

    def get_loss(
            self,
            pred_z,
            student_ability,
            question_difficulty,
            label,
            epsilon = 1e-6,
        ):
        label_1d = label.view(-1)
        pred_z_1d = pred_z.view(-1)
        student_ability_1d = student_ability.view(-1)
        question_difficulty_1d = question_difficulty.view(-1)

        # remove missing data
        index = torch.where(label_1d != -1)[0]

        filtered_label = torch.gather(label_1d, 0, index)
        filtered_z = torch.gather(pred_z_1d, 0, index)
        filtered_pred = torch.sigmoid(filtered_z)

        # get prediction probability from logit
        clipped_filtered_pred = torch.clamp(
            filtered_pred,
            epsilon,
            1. - epsilon,
        )
        filtered_logits = torch.log(
            clipped_filtered_pred / (1. - clipped_filtered_pred),
        )

        loss = F.binary_cross_entropy_with_logits(filtered_logits, filtered_label)
        pred_labels = filtered_pred.round()
        accuracy = pl_accuracy(pred_labels, filtered_label)
        auroc = pl_auroc(filtered_pred, filtered_label)
        return loss, accuracy, auroc

    def training_step(self, batch, batch_idx):
        index, response, problem_id, mask = batch
        response = response.to(self.device)
        problem_id = problem_id.to(self.device)
        mask = mask.to(self.device)
        pred_zs, student_abilities, question_difficulties = self(problem_id, response)
        loss, accuracy, auroc = self.get_loss(pred_zs, student_abilities, question_difficulties,
                                              response)
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        self.log('train_auroc', auroc)
        return loss

    def validation_step(self, batch, idx):
        return self.test_step(batch, idx, 'val')

    def test_step(self, batch, batch_idx, prefix='test', log=True):
        with torch.no_grad():
            index, response, problem_id, mask = batch
            response = response.to(self.device)
            problem_id = problem_id.to(self.device)
            mask = mask.to(self.device)
            pred_zs, student_abilities, question_difficulties = self(problem_id, response)
            loss, accuracy, auroc = self.get_loss(pred_zs, student_abilities, question_difficulties,
                                                  response)
            metrics = { f'{prefix}_loss': loss,
                        f'{prefix}_accuracy': accuracy,
                        f'{prefix}_auroc': auroc }

            self.log_dict(metrics)
            return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class DKVMN(nn.Module):
    """Adapted from the TensorFlow implementation.
    https://github.com/yjhong89/DKVMN/blob/master/model.py
    """

    def __init__(
            self,
            memory_size,
            memory_key_state_dim,
            memory_value_state_dim,
            init_memory_key,
            init_memory_value,
        ):
        super().__init__()
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.key = DKVMN_Memory(
            self.memory_size,
            self.memory_key_state_dim,
        )
        self.value = DKVMN_Memory(
            self.memory_size,
            self.memory_value_state_dim,
        )
        self.memory_key = init_memory_key
        self.memory_value = init_memory_value

    def attention(self, q_embedded):
        correlation_weight = self.key.cor_weight(
            q_embedded,
            self.memory_key,
        )
        return correlation_weight

    def read(self, c_weight):
        read_content = self.value.read(
            self.memory_value,
            c_weight,
        )
        return read_content

    def write(self, c_weight, qa_embedded):
        batch_size = c_weight.size(0)
        memory_value = self.value.write(
            self.memory_value,
            c_weight,
            qa_embedded,
        )
        self.memory_value = memory_value.detach()
        return memory_value


class DKVMN_Memory(nn.Module):
    """
    https://github.com/yjhong89/DKVMN/blob/master/memory.py
    """

    def __init__(self, memory_size, memory_state_dim):
        super().__init__()

        self.erase_linear = nn.Linear(memory_state_dim, memory_state_dim)
        self.add_linear = nn.Linear(memory_state_dim, memory_state_dim)

        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim

    def cor_weight(self, embedded, key_matrix):
        """
        embedded : (batch size, memory_state_dim)
        key_matrix : (memory_size, memory_state_dim)
        """
        # (batch_size, memory_size)
        embedding_result = embedded @ key_matrix.t()
        correlation_weight = torch.softmax(embedding_result, dim=1)
        return correlation_weight

    def read(self, value_matrix, correlation_weight):
        """
        value_matrix: (batch_size, memory_size, memory_state_dim)
        correlation_weight: (batch_size, memory_size)
        """
        batch_size = value_matrix.size(0)
        vmtx_reshaped = value_matrix.view(
            batch_size * self.memory_size,
            self.memory_state_dim,
        )
        cw_reshaped = correlation_weight.view(
            batch_size * self.memory_size,
            1
        )
        rc = vmtx_reshaped * cw_reshaped
        read_content = rc.view(
            batch_size,
            self.memory_size,
            self.memory_state_dim,
        )
        read_content = torch.sum(read_content, dim=1)

        return read_content

    def write(self, value_matrix, correlation_weight, qa_embedded):
        """
        value_matrix: (batch_size, memory_size, memory_state_dim)
        correlation_weight: (batch_size, memory_size)
        qa_embedded: (batch_size, memory_state_dim)
        """
        batch_size = value_matrix.size(0)

        erase_vector = self.erase_linear(qa_embedded)
        # (batch_size, memory_state_dim)
        erase_signal = torch.sigmoid(erase_vector)

        add_vector = self.add_linear(qa_embedded)
        # (batch_size, memory_state_dim)
        add_signal = torch.tanh(add_vector)

        erase_reshaped = erase_signal.view(
            batch_size,
            1,
            self.memory_state_dim,
        )
        cw_reshaped = correlation_weight.view(
            batch_size,
            self.memory_size,
            1,
        )
        erase_mul = erase_reshaped * cw_reshaped
        # (batch_size, memory_size, memory_state_dim)
        erase = value_matrix * (1 - erase_mul)

        # (batch_size, 1, memory_state_dim)
        add_reshaped = add_signal.view(
            batch_size,
            1,
            self.memory_state_dim,
        )
        add_mul = add_reshaped * cw_reshaped

        new_memory = erase + add_mul
        # (batch_size, memory_size, memory_state_dim)
        return new_memory

def split_train_val_test(d, frac_train, frac_val, split_seed=0):
    idx = list(range(len(d)))
    n_train = int(frac_train * len(d))
    n_val = int(frac_val * len(d))
    random.seed(split_seed)
    random.shuffle(idx)
    train_idx, val_idx, test_idx = (idx[:n_train],
                                    idx[n_train:n_train+n_val],
                                    idx[n_train+n_val:])
    print(n_train, 'train,', n_val, 'val,', len(idx) - n_train - n_val, 'test')
    return Subset(d, train_idx), Subset(d, val_idx), Subset(d, test_idx)

def evaluate(model, test_dataloader):
    metrics = collections.defaultdict(float)
    n = 0
    for i, batch in enumerate(test_dataloader):
        n_i = batch[3].sum()
        n += n_i
        m = model.test_step(batch, i)
        for k, v in m.items():
            metrics[k] += n_i * v

    for k, v in m.items():
        metrics[k] /= n

    return { **metrics, 'n': n }

def run_experiments(config):
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 100)

    gpus = GPUtil.getAvailable(order='memory', maxLoad=0.3, maxMemory=0.2)[:1] or None
    device = torch.device(gpus[0] if gpus else 'cpu')
    print('Using device', device)

    run = wandb.init(reinit=True, config=config)

    d = dataset.CognitiveTutorDataset(config['dataset'])

    print(d.n_problems, 'problems.')

    needs_training = True

    if config['model']['type'] == 'DKVMN':
        irt = DKVMN_IRT(config['model'], device, batch_size, d.n_problems, 100)
    elif config['model']['type'] == 'DKT':
        irt = DKT(config['model'], d.n_problems)
    else:
        irt = EKT(config['model'], d.n_problems)
        needs_training = False

    irt.to(device)

    if config.get('initialize_embeddings'):
        emb_model = LearnerValueFunction.load(config['embeddings_model'], map_location=device)
        emb_model.to(device)

        print('Embedding problems...')
        with torch.no_grad():
            problem_embeddings = []
            for b in batched(d.problems, batch_size):
                problem_embeddings.append(emb_model.embed_problems(b))
            problem_embeddings = torch.cat(problem_embeddings)
            print('Embeddings dimension:', problem_embeddings.shape)
            if config.get('normalize_embeddings'):
                problem_embeddings /= (problem_embeddings**2).sum(dim=1).sqrt()[:, None]
            alpha = config.get('embeddings_alpha', 1.0)
            irt.q_embed_matrix = nn.Embedding.from_pretrained(alpha * problem_embeddings)
            print('Done!')

            if config.get('dump_similarities'):
                torch.save({
                    'embeddings': irt.q_embed_matrix.weight,
                    'similarity': irt.q_embed_matrix.weight.matmul(irt.q_embed_matrix.weight.T),
                    }, config['dump_similarities'])
                print('Dumped similarity matrix.')

    print('Average embedding L2-norm:',
          (irt.q_embed_matrix.weight**2).sum(dim=1).sqrt().mean())

    irt.q_embed_matrix.weight.requires_grad = not config.get('freeze_embeddings', False)

    training_set, val_set, test_set = split_train_val_test(
            d,
            config['training_fraction'],
            config['val_fraction'],
            config['split_seed'])

    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    trainer = pl.Trainer(gpus=gpus,
                         logger=pl.loggers.wandb.WandbLogger(config.get('name', 'DeepIRT')),
                         max_epochs=epochs)

    if needs_training:
        trainer.fit(irt, train_dataloader, val_dataloader)
        trainer.test(test_dataloaders=test_dataloader)

    results = evaluate(irt, test_dataloader)
    print('Manually evaluating on test set:', results)

    run.finish()

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file')

    opt = parser.parse_args()

    config = json.load(open(opt.config))

    run_experiments(config)
