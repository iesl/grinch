"""
Copyright (C) 2019-2020 University of Massachusetts Amherst.
This file is part of “grinch”
http://github.com/iesl/grinch
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import collections
import os

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from absl import logging
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from grinch.agglom import Agglom
from grinch.eval_pw_f1 import eval_micro_pw_f1


def compute_margin_loss(grinch, samples, margin):
    pos, neg = grinch_sims(grinch, samples)
    return margin_loss(pos, neg, margin)


def margin_loss(positive_scores, negative_scores, margin):
    logging.log_first_n(logging.INFO, '[margin_loss] positive %s | negative %s | margin %s', 10,
                        str(positive_scores.shape), str(negative_scores.shape), margin)
    pos_minus_neg = positive_scores - negative_scores - margin
    labels = torch.ones_like(pos_minus_neg)
    res = F.binary_cross_entropy_with_logits(pos_minus_neg, labels, reduction='mean')
    return res


def build_random_samples_from_points(grinch, points, labels, num_samples, num_negatives):
    def clusters2dict(assgn):
        d = collections.defaultdict(list)
        for idx, c in enumerate(assgn):
            d[c].append(idx)
        return d

    assignment_dict = clusters2dict(labels)
    filtered = [y for x, y in assignment_dict.items() if len(y) > 2]
    samples = np.zeros((num_samples, 3), np.int32)
    for idx in range(num_samples):
        random_clusters = np.random.choice(len(filtered), 2, replace=False)
        random_cluster = random_clusters[0]
        pos_pair = np.random.choice(filtered[random_cluster], 2, replace=False)
        samples[idx, 0:2] = pos_pair
        neg_cluster = random_clusters[1]
        neg_idx = np.random.choice(filtered[neg_cluster])
        samples[idx, 2] = neg_idx

    return samples


def grinch_sims(grinch, samples):
    pos = grinch.pw_sim_torch(samples[:, 0], samples[:, 1])
    neg = grinch.pw_sim_torch(samples[:, 0], samples[:, 2])
    return pos, neg


def grinch_sims_look(grinch, samples, points):
    record_pos = dict()
    record_neg = dict()
    pos = grinch.pw_sim_torch(samples[:, 0], samples[:, 1], record_pos)
    neg = grinch.pw_sim_torch(samples[:, 0], samples[:, 2], record_neg)

    for i in range(pos.shape[0]):
        logging.info("Example i=%s" % i)
        logging.info('Src: %s', points[samples[i, 0]].pretty_tsv())
        logging.info('Pos: %s', points[samples[i, 1]].pretty_tsv())
        logging.info('Neg: %s', points[samples[i, 2]].pretty_tsv())
        for k in record_pos.keys():
            logging.info('%s\t%s\t%s', k, record_pos[k][i], record_neg[k][i])

    return pos, neg


class Trainer(object):
    """Train a linear+rule model."""

    def __init__(self, outdir, model, encoding_model, grinch, pids, labels, points, dev_data, num_samples,
                 num_negatives, batch_size, dev_every, epochs, lr, weight_decay, margin, num_thresholds=25,
                 save_every=100, log_every=100, max_dev_size=5000, max_dev_canopies=50):

        logging.info('building random samples...')
        self.outdir = outdir
        self.samples = build_random_samples_from_points(grinch, points, labels, num_samples, num_negatives)
        self.grinch = grinch
        self.points = points
        self.pids = pids
        self.batch_size = batch_size
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.margin = margin
        self.epochs = epochs
        self.dev_every = dev_every
        self.dev_data = dev_data
        self.global_step = 0
        self.model = model
        self.encoding_model = encoding_model
        self.num_thresholds = num_thresholds
        self.save_every = save_every
        self.best_f1 = 0.0
        self.max_dev_size = max_dev_size
        self.max_dev_canopies = max_dev_canopies
        self.filter_dev_dataset()

    def filter_dev_dataset(self):
        self.orig_dev_data = self.dev_data
        self.dev_data = [x for x in self.orig_dev_data.items() if len(x[1]) < self.max_dev_size]
        self.dev_data = sorted(self.dev_data, key=lambda x: len([y for y in x[1][1] if y != '-1' and y != -1]),
                               reverse=True)
        self.dev_data = dict(self.dev_data[:self.max_dev_canopies])
        with open(self.outdir + '/dev_gold.tsv', 'w') as fout:
            for idx, (dataset_name, dataset) in enumerate(self.dev_data.items()):
                pids, lbls, records, features = dataset[0], dataset[1], dataset[2], dataset[3]
                for i in range(len(pids)):
                    fout.write('%s\t%s\n' % (pids[i], lbls[i]))

    def save(self, name=None):
        torch.save(self.model, os.path.join(self.outdir, 'model-%s.torch' % (self.global_step if name else name)))

    def print_model_weights(self):
        logging.info('------- Model Weights -------')
        for i in range(len(self.encoding_model.feature_list)):
            logging.info('%s %s', self.encoding_model.feature_list[i].name,
                         str(self.model.weight_for(self.encoding_model.feature_list[i].name)))
        logging.info('------- END Weights -------')

    def print_small_examples(self):
        logging.info('------- Examples -------')
        x = np.random.choice(len(self.samples), 5)
        grinch_sims_look(self.grinch, self.samples[x, :], self.points)
        logging.info('------- END -------')

    def get_thresholds(self, trees, T):
        logging.info('getting thresholds from %s trees', len(trees))
        scores = []
        for grinch in tqdm(trees, 'getting thresholds'):
            scores.append(grinch.all_thresholds())
        scores = np.concatenate(scores).flatten()
        scores = np.expand_dims(scores, 1)
        logging.info('thresholds - %s ', scores.shape)
        logging.info('running kmeans with k=%s', T)
        km = MiniBatchKMeans(n_clusters=T)
        km.fit(scores)
        return km.cluster_centers_

    def train(self):
        for epoch in range(self.epochs):
            for i in range(0, len(self.samples), self.batch_size):
                self.opt.zero_grad()
                batch_samples = self.samples[i:(i + self.batch_size)]
                loss = compute_margin_loss(self.grinch, batch_samples, self.margin)
                loss.backward()
                self.opt.step()
                logging.log_every_n(logging.INFO, 'train loss: loss %s @ batch %s', 100, loss.detach().cpu().numpy(), i)
                if self.global_step % 100 == 0:
                    # wandb.log({'train_loss': loss.detach().numpy(), 'global_step': self.global_step})
                    self.print_model_weights()
                    self.print_small_examples()
                if self.global_step > 0 and self.global_step % self.dev_every == 0 and self.dev_data is not None:
                    self.dev_eval(self.dev_data)
                if self.global_step > 0 and self.global_step % self.save_every == 0:
                    self.save()
                self.global_step += 1

    def dev_eval(self, datasets):
        logging.info('dev eval using %s datasets', len(datasets))
        trees = []
        gold_clustering = []
        dataset_names = []
        for idx, (dataset_name, dataset) in enumerate(datasets.items()):
            pids, lbls, records, features = dataset[0], dataset[1], dataset[2], dataset[3]
            logging.info('Running on dev dataset %s of %s | %s with %s points', idx, len(datasets), dataset_name,
                         len(pids))
            if len(pids) > 0:
                grinch = Agglom(self.grinch.model, features, num_points=len(pids))
                grinch.build_dendrogram_hac()
                trees.append(grinch)
                gold_clustering.extend(lbls)
                dataset_names.append(dataset_name)
        eval_ids = [i for i in range(len(gold_clustering)) if gold_clustering[i] != '-1']
        thresholds = np.sort(np.squeeze(self.get_thresholds(trees, self.num_thresholds)))
        scores_per_threshold = []
        os.makedirs(os.path.join(self.outdir, 'dev'), exist_ok=True)
        dev_out_f = open(os.path.join(self.outdir, 'dev', 'dev_%s.tsv' % self.global_step), 'w')
        for thres in thresholds:
            pred_clustering = []
            for idx, t in enumerate(trees):
                fc = t.flat_clustering(thres)
                pred_clustering.extend(['%s-%s' % (dataset_names[idx], x) for x in fc])
            metrics = eval_micro_pw_f1([pred_clustering[x] for x in eval_ids], [gold_clustering[x] for x in eval_ids])
            scores_per_threshold.append(metrics)
            logging.info('[dev] threshold %s | %s', thres, "|".join(['%s=%s' % (k, v) for k, v in metrics.items()]))

        arg_best_f1 = max([x for x in range(len(scores_per_threshold))],
                          key=lambda x: scores_per_threshold[x]['micro_pw_f1'])
        for idx, t in enumerate(trees):
            dataset = datasets[dataset_names[idx]]
            pids, lbls, records, features = dataset[0], dataset[1], dataset[2], dataset[3]
            fc = t.flat_clustering(thresholds[arg_best_f1])
            for j in range(len(records)):
                dev_out_f.write("%s\n" % records[j].pretty_tsv('%s-%s' % (dataset_names[idx], fc[j]), lbls[j]))
        metrics = scores_per_threshold[arg_best_f1]
        logging.info('[dev] best threshold %s | %s', thresholds[arg_best_f1],
                     "|".join(['%s=%s' % (k, v) for k, v in metrics.items()]))
        dev_out_f.close()
        if metrics['micro_pw_f1'] > self.best_f1:
            logging.info('new best f1 %s > %s', metrics['micro_pw_f1'], self.best_f1)
            self.model.aux['threshold'] = thresholds[arg_best_f1]
            self.best_f1 = metrics['micro_pw_f1']
            self.save('best')
        return thresholds[arg_best_f1], metrics
