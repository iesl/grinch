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

import time

import numpy as np
import torch
import torch.nn
from absl import logging
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances, paired_distances

from grinch.features import FeatCalc

logging.set_verbosity(logging.INFO)


class Agglom(object):
    """ Hierarchical agglomerative clustering with learned linear model w/ rules."""

    def __init__(self, model, features, num_points, min_allowable_sim=-20000.0):
        """ Constructor for Agglom.

        :param model: Pairwise linear scoring function
        :param features: features used in linear model.
        :param num_points: number of points that we are clustering
        :param min_allowable_sim: minimum allowable similarity (used for rule-based models)
        """

        self.model = model
        self.features = features
        self.num_points = num_points

        logging.info('Using len(features)=%s', len(features))
        self.dense_features = [(fn, is_dense, dim, feat_mat, feat_calc, centroid_type)
                               for fn, is_dense, dim, feat_mat, feat_calc, centroid_type in features if is_dense]
        self.sparse_features = [(fn, is_dense, dim, feat_mat, feat_calc, centroid_type)
                                for fn, is_dense, dim, feat_mat, feat_calc, centroid_type in features if not is_dense]

        logging.info('%s dense features: %s', len(self.dense_features), ', '.join([x[0] for x in self.dense_features]))
        logging.info('%s sparse features: %s', len(self.sparse_features),
                     ', '.join([x[0] for x in self.sparse_features]))

        self.min_allowable_sim = min_allowable_sim

        self.dense_point_features = []
        self.sparse_point_features = []
        self.dense_feature_id = dict()
        for idx, (fn, is_dense, dim, feat_mat, _, _) in enumerate(self.dense_features):
            self.dense_feature_id[fn] = idx
            self.dense_point_features.append(feat_mat)

        self.sparse_feature_id = dict()
        for idx, (fn, is_dense, dim, feat_mat, _, _) in enumerate(self.sparse_features):
            self.sparse_feature_id[fn] = idx
            # analyze_sparsity(feat_mat, fn)
            self.sparse_point_features.append(feat_mat)

        self.dists = None
        self.Z = None

    def all_thresholds(self):
        """All possible ways to cut a tree."""
        return self.Z[:, 2]

    def flat_clustering(self, threshold):
        """A flat clustering extracted from the tree by cutting at the given distance threshold.

        :param threshold: distance threshold
        :return: res - N element array such that res[i] is the cluster id of the ith point.
        """
        return fcluster(self.Z, threshold, criterion='distance')

    def build_dendrogram_hac(self):
        """Run HAC inference."""
        st_sims = time.time()
        sims = self.csim_multi_feature_knn_batched(np.arange(self.num_points), np.arange(self.num_points))
        en_sims = time.time()
        logging.info('Time to compute sims: %s', en_sims - st_sims)
        logging.info('Finished batched similarities!')
        st_prep = time.time()
        pos_sim = np.maximum(sims, self.min_allowable_sim) - np.minimum(0.0, self.min_allowable_sim)
        del sims
        dists = 1 / (1 + pos_sim)
        del pos_sim
        dists = (dists + dists.T) / 2.0
        np.fill_diagonal(dists, 0.0)
        dists = squareform(dists)
        en_prep = time.time()
        logging.info('Time to compute sims: %s', en_prep - st_prep)
        logging.info('Finished preparing distances!')
        logging.info('Running hac')
        st_linkage = time.time()
        Z = linkage(dists, method='average')
        del dists
        self.Z = Z
        en_linkage = time.time()
        logging.info('Time to run HAC: %s', en_linkage - st_linkage)
        logging.info('Finished hac!')

    def csim_multi_feature_knn_batched(self, i, j, batch_size=7000):
        """Compute the pairwise similarities between all pairs of i's and j's."""
        len_i = len(i)
        len_j = len(j)
        should_log = len_i > 3 * batch_size
        if should_log:
            logging.info('[csim_multi_feature_knn_batched] %s by %s', len_i, len_j)
        res = np.zeros((len_i, len_j), dtype=np.float32)
        i_batches = [x for x in range(0, len_i, batch_size)]
        st = time.time()
        for idx, ii in enumerate(i_batches):
            if should_log:
                logging.info('[csim_multi_feature_knn_batched] %s by %s - %s of %s in % seconds', len_i, len_j, idx,
                             len(i_batches) - 1, time.time() - st)
            for jj in range(0, len_j, batch_size):
                istart = ii
                jstart = jj
                iend = min(len_i, ii + batch_size)
                jend = min(len_j, jj + batch_size)
                res[istart:iend, jstart:jend] = self.csim_multi_feature_knn(i[istart:iend], j[jstart:jend]).astype(
                    np.float32)
        logging.info('[csim_multi_feature_knn_batched] DONE! %s by %s - %s of %s in % seconds', len_i, len_j,
                     len(i_batches) - 1,
                     len(i_batches) - 1, time.time() - st)
        return res

    def csim_multi_feature_knn(self, i, j):
        return self.csim_multi_feature_knn_torch(i, j).detach().numpy()

    def csim_multi_feature_knn_torch(self, i, j, record_dict=None):
        len_i = len(i)
        len_j = len(j)
        s = torch.zeros((len_i, len_j), dtype=torch.float32)
        for idx in range(len(self.dense_features)):
            w, b = self.model.weight_for(self.dense_features[idx][0])
            lhs = self.dense_features[idx][3][i]  # Grab the original feature matrix for i
            rhs = self.dense_features[idx][3][j]  # Grab the original feature matrix for j
            if self.dense_features[idx][4] == FeatCalc.L2:
                feat_score = w * self.c_l2dist_feature_dense_knn(idx, lhs, rhs)
            elif self.dense_features[idx][4] == FeatCalc.DOT:
                feat_score = w * self.c_dot_feature_dense_knn(idx, lhs, rhs)
            elif self.dense_features[idx][4] == FeatCalc.L2_gt_one:
                feat_score = w * self.c_l2dist_gt_one_feature_dense_knn(idx, lhs, rhs)
            elif self.dense_features[idx][4] == FeatCalc.NO_MATCH:
                feat_score = w * self.c_no_match_feature_dense_knn(idx, lhs, rhs)
            else:
                raise Exception('no feature %s' % self.dense_features[idx][4])
            if b is not None:
                feat_score += b
            if record_dict is not None:
                record_dict[self.dense_features[idx][0]] = feat_score
            s += feat_score
        for idx in range(len(self.sparse_features)):
            w, b = self.model.weight_for(self.sparse_features[idx][0])
            lhs = self.sparse_features[idx][3][i]  # Grab the original feature matrix for i
            rhs = self.sparse_features[idx][3][j]  # Grab the original feature matrix for j
            if self.sparse_features[idx][4] == FeatCalc.L2:
                feat_score = w * self.c_l2dist_feature_sparse_knn(idx, lhs, rhs)
            elif self.sparse_features[idx][4] == FeatCalc.DOT:
                feat_score = w * self.c_dot_feature_sparse_knn(idx, lhs, rhs)
            elif self.sparse_features[idx][4] == FeatCalc.L2_gt_one:
                feat_score = w * self.c_l2dist_gt_one_feature_sparse_knn(idx, lhs, rhs)
            else:
                raise Exception('no feature %s' % self.sparse_features[idx][4])
            if b is not None:
                feat_score += b
            if record_dict is not None:
                record_dict[self.sparse_features[idx][0]] = feat_score
            s += feat_score
        return s

    def pw_sim_torch(self, i, j, record_dict=None):
        assert len(i) == len(j)
        s = torch.zeros(len(i), dtype=torch.float32)
        for idx in range(len(self.dense_features)):
            w, b = self.model.weight_for(self.dense_features[idx][0])
            lhs = self.dense_features[idx][3][i]  # Grab the original feature matrix for i
            rhs = self.dense_features[idx][3][j]  # Grab the original feature matrix for j
            if self.dense_features[idx][4] == FeatCalc.L2:
                feat_score = w * self.p_l2dist_feature_dense(idx, lhs, rhs)
            elif self.dense_features[idx][4] == FeatCalc.DOT:
                feat_score = w * self.p_dot_feature_dense(idx, lhs, rhs)
            elif self.dense_features[idx][4] == FeatCalc.L2_gt_one:
                feat_score = w * self.p_l2dist_gt_one_feature_dense(idx, lhs, rhs)
            elif self.dense_features[idx][4] == FeatCalc.NO_MATCH:
                feat_score = w * self.p_no_match_feature_dense(idx, lhs, rhs)
            else:
                raise Exception('no feature %s' % self.dense_features[idx][4])
            if b is not None:
                feat_score += b
            if record_dict is not None:
                record_dict[self.dense_features[idx][0]] = feat_score
            s += feat_score
        for idx in range(len(self.sparse_features)):
            w, b = self.model.weight_for(self.sparse_features[idx][0])
            lhs = self.sparse_features[idx][3][i]  # Grab the original feature matrix for i
            rhs = self.sparse_features[idx][3][j]  # Grab the original feature matrix for j
            if self.sparse_features[idx][4] == FeatCalc.L2:
                feat_score = w * self.p_l2dist_feature_sparse(idx, lhs, rhs)
            elif self.sparse_features[idx][4] == FeatCalc.DOT:
                feat_score = w * self.p_dot_feature_sparse(idx, lhs, rhs)[:, 0]
            elif self.sparse_features[idx][4] == FeatCalc.L2_gt_one:
                feat_score = w * self.p_l2dist_gt_one_feature_sparse(idx, lhs, rhs)
            else:
                raise Exception('no feature %s' % self.sparse_features[idx][4])
            if b is not None:
                feat_score += b
            if record_dict is not None:
                record_dict[self.sparse_features[idx][0]] = feat_score
            s += feat_score
        return s

    def p_l2dist_feature_dense(self, idx, c_i, c_j):
        res = paired_distances(c_i, c_j, metric='euclidean') ** 2
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_feature_dense(self, idx, c_i, c_j):
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_feature_dense_knn(self, idx, c_i, c_j):
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2
        return torch.from_numpy(res.astype(np.float32))

    def p_l2dist_gt_one_feature_dense(self, idx, c_i, c_j):
        res = (paired_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2) > 1
        return torch.from_numpy(res.astype(np.float32))

    def p_no_match_feature_dense(self, idx, c_i, c_j):
        res = np.squeeze(np.logical_and(np.logical_and(c_i != c_j, c_i != -1), c_j != -1), 1)
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_gt_one_feature_dense(self, idx, c_i, c_j):
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2 > 1
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_gt_one_feature_dense_knn(self, idx, c_i, c_j):
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2 > 1
        return torch.from_numpy(res.astype(np.float32))

    def c_no_match_feature_dense_knn(self, idx, c_i, c_j):
        c_jT = c_j.T
        res = np.logical_and(np.logical_and(c_i != c_jT, c_i != -1), c_jT != -1)
        return torch.from_numpy(res.astype(np.float32))

    def c_l1_dense_knn(self, idx, c_i, c_j):
        return cdist(c_i, c_j, metric='cityblock')

    def c_ratio_dense_knn(self, idx, c_i, c_j):
        res = c_i / c_j.T
        return res

    def p_dot_feature_dense(self, idx, c_i, c_j):
        res = np.sum(c_i * c_j, axis=1).astype(np.float32)
        return torch.from_numpy(res.astype(np.float32))

    def c_dot_feature_dense(self, idx, c_i, c_j):
        res = np.matmul(c_i, c_j.T)
        return torch.from_numpy(res.astype(np.float32))

    def c_dot_feature_dense_knn(self, idx, c_i, c_j):
        res = np.matmul(c_i, c_j.T)
        return torch.from_numpy(res.astype(np.float32))

    def p_l2dist_feature_sparse(self, idx, c_i, c_j):
        res = paired_distances(c_i, c_j, metric='euclidean') ** 2
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_feature_sparse(self, idx, c_i, c_j):
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_feature_sparse_knn(self, idx, c_i, c_j):
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2
        return torch.from_numpy(res.astype(np.float32))

    def p_l2dist_gt_one_feature_sparse(self, idx, c_i, c_j):
        res = paired_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2 > 1
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_gt_one_feature_sparse(self, idx, c_i, c_j):
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2 > 1
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_gt_one_feature_sparse_knn(self, idx, c_i, c_j):
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2 > 1
        return torch.from_numpy(res.astype(np.float32))

    def p_dot_feature_sparse(self, idx, c_i, c_j):
        res = c_i.multiply(c_j).sum(axis=1).A.astype(np.float32)
        return torch.from_numpy(res.astype(np.float32))

    def c_dot_feature_sparse(self, idx, c_i, c_j):
        res = (c_i @ c_j.T).todense().A
        return torch.from_numpy(res.astype(np.float32))

    def c_dot_feature_sparse_knn(self, idx, c_i, c_j):
        res = (c_i @ c_j.T).todense().A
        return torch.from_numpy(res.astype(np.float32))
