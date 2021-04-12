import pickle
import time

import numpy as np
import scipy.sparse.linalg
import torch
import torch.nn
from absl import logging
from sklearn.metrics.pairwise import pairwise_distances, paired_distances

from grinch.features import FeatCalc, CentroidType
from grinch.grinch_alg import Grinch

logging.set_verbosity(logging.INFO)


class MultiFeatureGrinch(Grinch):

    def __init__(self, features, num_points, dim=None, rotate_cap=100, graft_cap=100, norm='l2', sim='dot',
                 max_num_points=None):
        """

        :param features: (fn, is_dense, dim, feat_mat)
        :param points:
        :param rotate_cap:
        :param graft_cap:
        :param norm:
        :param sim:
        """
        super(MultiFeatureGrinch, self).__init__(num_points=num_points, dim=dim, rotate_cap=rotate_cap,
                                                 graft_cap=graft_cap, norm=norm, sim=sim, max_num_points=max_num_points)

        self.features = features

        self.dense_features = [(fn, is_dense, dim, feat_mat, feat_calc, centroid_type)
                               for fn, is_dense, dim, feat_mat, feat_calc, centroid_type in features if is_dense]
        self.sparse_features = [(fn, is_dense, dim, feat_mat, feat_calc, centroid_type)
                                for fn, is_dense, dim, feat_mat, feat_calc, centroid_type in features if not is_dense]

        logging.info('%s dense features', len(self.dense_features))
        logging.info('%s sparse features', len(self.sparse_features))

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

        self.dense_centroids = [[] for _ in range(len(self.dense_feature_id))]
        self.dense_sums = [[] for _ in range(len(self.dense_feature_id))]

        self.sparse_centroids = [[] for _ in range(len(self.sparse_feature_id))]
        self.sparse_sums = [[] for _ in range(len(self.sparse_feature_id))]
        self.init_features()
        self.csim = self.csim_multi_feature
        self.compute_centroid = self.compute_centroid_multi_feature
        self.points_set = False


    def clear_node_features(self):
        self.dense_centroids = [[] for _ in range(len(self.dense_feature_id))]
        self.dense_sums = [[] for _ in range(len(self.dense_feature_id))]

        self.sparse_centroids = [[] for _ in range(len(self.sparse_feature_id))]
        self.sparse_sums = [[] for _ in range(len(self.sparse_feature_id))]

        for x in self.all_valid_internal_nodes():
            self.needs_update_model[x] = True
            self.needs_update_desc[x] = True
            self.descendants[x] = []

    def update_and_insert(self, i_features, pids=None):

        # keep track of pids
        if pids is not None:
            assert self.pids is not None
            self.pids.extend(pids)

        if self.num_points != self.point_counter:
            logging.info('something is wrong, point_counter != num_points')
            self.point_counter = self.num_points

        logging.info('[update_and_insert] starting now.....')
        s = time.time()

        logging.info('max_num_points %s | current num_points %s', self.max_num_points, self.num_points)
        num_points, prev_num_points = self.update_features(i_features)

        if self.points_set is False:
            logging.info('grinch points not set, setting now for all points.....')
            self.set_points(np.arange(self.num_points))

        for idx in range(len(self.dense_point_features)):
            logging.info('dense_point_features %s', str(self.dense_point_features[idx].shape))
            logging.info('dense_sums %s', str(self.dense_sums[idx].shape))
            logging.info('dense_centroids %s', str(self.dense_centroids[idx].shape))

        for idx in range(len(self.sparse_point_features)):
            logging.info('sparse_point_features %s', str(self.sparse_point_features[idx].shape))
            logging.info('sparse_sums %s', str(len(self.sparse_sums[idx])))
            logging.info('sparse_centroids %s', str(len(self.sparse_centroids[idx])))

        logging.info('max_num_points %s | current num_points %s | adding %s', self.max_num_points, self.num_points, num_points)
        for i in range(prev_num_points, prev_num_points + num_points):
            logging.info('[update_and_insert] inserting %s - elapsed %s', i, time.time() - s)
            self.insert(i)
            self.num_points += 1
        logging.info('max_num_points %s | current num_points %s | adding %s', self.max_num_points, self.num_points,
                     num_points)
        logging.debug('[update_and_insert] starting now.....done! Elapsed %s', time.time() - s)

    def update_features(self, i_features):
        logging.debug('updating %s features', len(i_features))

        dense_features = [(fn, is_dense, dim, feat_mat, feat_calc, centroid_type)
                          for fn, is_dense, dim, feat_mat, feat_calc, centroid_type in i_features if is_dense]
        sparse_features = [(fn, is_dense, dim, feat_mat, feat_calc, centroid_type)
                           for fn, is_dense, dim, feat_mat, feat_calc, centroid_type in i_features if not is_dense]

        logging.debug('%s dense features', len(dense_features))
        logging.debug('%s sparse features', len(sparse_features))

        dense_feature_id = dict()
        for idx, (fn, is_dense, dim, feat_mat, _, _) in enumerate(dense_features):
            dense_feature_id[fn] = idx
            num_points = feat_mat.shape[0]
            prev_num_points = self.dense_point_features[idx].shape[0]
            logging.debug('[old] dense_point_features[%s].shape = %s', idx, str(self.dense_point_features[idx].shape))
            logging.debug('[old] feat_mat.shape = %s', str(feat_mat.shape))
            self.dense_point_features[idx] = np.vstack([self.dense_point_features[idx], feat_mat])
            logging.debug('[new] dense_point_features[%s].shape = %s', idx, str(self.dense_point_features[idx].shape))

        sparse_feature_id = dict()
        for idx, (fn, is_dense, dim, feat_mat, _, _) in enumerate(sparse_features):
            sparse_feature_id[fn] = idx
            num_points = feat_mat.shape[0]
            prev_num_points = self.sparse_point_features[idx].shape[0]
            logging.debug('[old] sparse_point_features[%s].shape = %s', idx, str(self.sparse_point_features[idx].shape))
            logging.debug('[old] feat_mat.shape = %s', str(feat_mat.shape))
            self.sparse_point_features[idx] = scipy.sparse.vstack([self.sparse_point_features[idx], feat_mat])
            logging.debug('[new] sparse_point_features[%s].shape = %s', idx, str(self.sparse_point_features[idx].shape))

        return num_points, prev_num_points

    def add_pt(self, i):
        logging.log_every_n(logging.debug, 'Adding point %s', 1000, i)
        for idx, mat in enumerate(self.dense_point_features):
            # logging.log_every_n(logging.debug, 'Adding point %s - dense_point_features %s -> shape %s', 1000, i, idx, str(mat[i].shape))
            self.dense_sums[idx][i] = mat[i].copy()
        for idx, mat in enumerate(self.sparse_point_features):
            # logging.log_every_n(logging.debug, 'Adding point %s - sparse_point_features %s -> shape %s %s', 1000, i, idx, str(mat[i].shape), type(mat))
            self.sparse_sums[idx][i] = mat[i].copy()
        self.num_descendants[i] = 1
        self.point_counter += 1
        self.descendants[i].append(i)
        self.compute_centroid(i, False)
        self.new_node[i] = False

    def set_points(self, i):
        """

        Note: unlike add_pt this does not copy features..

        :param i:
        :return:
        """
        self.init_features()
        logging.log_every_n(logging.debug, 'Adding %s points', 1000, len(i))
        for idx, mat in enumerate(self.dense_point_features):
            logging.log_every_n(logging.debug, 'Adding point %s - dense_point_features %s -> shape %s', 1000, i, idx, str(mat[i].shape))
            self.dense_sums[idx][i] = mat[i]
            self.dense_centroids[idx][i] = mat[i]
        for idx, mat in enumerate(self.sparse_point_features):
            logging.log_every_n(logging.debug, 'Adding point %s - sparse_point_features %s -> shape %s %s', 1000, i, idx, str(mat[i].shape), type(mat))
            for ii in i:
                self.sparse_sums[idx][ii] = mat[ii]
                self.sparse_centroids[idx][ii] = mat[ii]

        self.points_set = True

    def from_scipy_z(self, Z, update=False, pids=None, canopies=None):

        self.pids = pids
        self.canopies = canopies
        logging.debug('running from_scipy_z')
        assert self.num_points == Z.shape[0] + 1
        should_log = self.num_points > 30000
        id_map = dict()
        self.next_node_id = self.max_num_points * 2
        self.point_counter = self.num_points-1

        def get_id(scipy_id):
            if scipy_id < self.num_points:
                logging.log_first_n(logging.DEBUG, 'scipy_id %s in id_map', 10, scipy_id)
                return scipy_id
            else:
                if scipy_id not in id_map:
                    id_map[scipy_id] = self.next_node_id
                    self.next_node_id += 1
                    logging.log_first_n(logging.DEBUG, 'scipy_id %s not in id_map, id_map[scipy_id]=%s, self.next_node_id', 10, scipy_id, id_map[scipy_id], self.next_node_id)

                return id_map[scipy_id]
        for i in range(Z.shape[0]):
            if should_log:
                logging.log_every_n(logging.debug, 'from scipy z, processed %s of %s', 1000, i, Z.shape[0])
            internal_id = get_id(i + self.num_points)
            self.parent[get_id(Z[i, 0].astype(np.int32))] = internal_id
            self.parent[get_id(Z[i, 1].astype(np.int32))] = internal_id
            self.children[internal_id] = [get_id(Z[i, 0].astype(np.int32)), get_id(Z[i, 1].astype(np.int32))]
            self.needs_update_desc[internal_id] = True
            self.needs_update_model[internal_id] = True
        logging.debug('root is %s', self.root())
        assert self.needs_update_model[self.root()]
        assert self.needs_update_desc[self.root()]
        self.descendants[0:self.num_points] = [[x] for x in range(self.num_points)]
        self.num_descendants[0:self.num_points] = [1 for x in range(self.num_points)]

        if update:
            self.update_desc(self.root())
            self.update(self.root())

    def grow_dense_feature(self, new_max_nodes, existing):
        to_append1 = np.zeros((new_max_nodes-self.max_nodes, existing[0].shape[1]), dtype=np.float32)
        to_append2 = np.zeros((new_max_nodes-self.max_nodes, existing[1].shape[1]), dtype=np.float32)
        new_df1 = np.vstack([existing[0], to_append1])
        new_df2 = np.vstack([existing[1], to_append2])
        return new_df1, new_df2

    def grow_sparse_feature(self, new_max_nodes, existing):
        to_append1 = [[] for _ in range(new_max_nodes - self.max_nodes)]
        to_append2 = [[] for _ in range(new_max_nodes - self.max_nodes)]
        new_sf1 = existing[0].extend(to_append1)
        new_sf2 = existing[1].extend(to_append2)
        return new_sf1, new_sf2

    def grow_features(self, new_max_nodes):
        for idx, (fn, is_dense, dim, feat_mat, _, _) in enumerate(self.dense_features):
            logging.debug('Initialize feature - name=%s, dense=%s, dim=%s, mat=%s', fn, is_dense, dim,
                         str(feat_mat.shape))
            assert self.dense_feature_id[fn] == idx
            c, s = self.grow_dense_feature(new_max_nodes, [self.dense_centroids[idx], self.dense_sums[idx]])
            self.dense_centroids[idx] = c
            self.dense_sums[idx] = s
            logging.debug(
                'Initialize feature - name=%s, dense=%s, dim=%s, mat=%s, self.dense_centroids[%s]=%s, self.dense_sums[%s]=%s',
                fn, is_dense, dim,
                str(feat_mat.shape), idx, str(self.dense_centroids[idx].shape), idx,
                str(self.dense_sums[idx].shape))

        for idx, (fn, is_dense, dim, feat_mat, _, _) in enumerate(self.sparse_features):
            logging.debug('Initialize feature - name=%s, dense=%s, dim=%s, mat=%s', fn, is_dense, dim,
                         str(feat_mat.shape))
            assert self.sparse_feature_id[fn] == idx
            c, s = self.grow_sparse_feature(new_max_nodes, [self.sparse_centroids[idx], self.sparse_sums[idx]])
            self.sparse_centroids[idx] = c
            self.sparse_sums[idx] = s

    def grow_if_necessary(self):
        if self.next_node_id >= self.max_nodes:
            logging.debug('resizing internal structures...')
            new_max_nodes = 2 * self.max_nodes
            logging.debug('new max nodes %s', new_max_nodes)
            self.grow_nodes(new_max_nodes)
            self.grow_features(new_max_nodes)
            self.max_nodes = new_max_nodes

    def init_dense_feature(self, dim):
        return np.zeros((self.max_nodes, dim), dtype=np.float32), np.zeros((self.max_nodes, dim), dtype=np.float32)

    # def init_sparse_feature(self, dim):
    #     return csr_matrix((self.max_num_points, dim), dtype=np.float32), csr_matrix((self.max_num_points, dim), dtype=np.float32)

    def init_sparse_feature(self, dim):
        return [[] for _ in range(self.max_nodes)], [[] for _ in range(self.max_nodes)]

    def init_features(self):
        for idx, (fn, is_dense, dim, feat_mat, _, _) in enumerate(self.dense_features):
            logging.debug('Initialize feature - name=%s, dense=%s, dim=%s, mat=%s', fn, is_dense, dim,
                         str(feat_mat.shape))
            self.dense_feature_id[fn] = idx
            c, s = self.init_dense_feature(dim)
            self.dense_centroids[idx] = s
            self.dense_sums[idx] = c
            logging.debug(
                'Initialize feature - name=%s, dense=%s, dim=%s, mat=%s, self.dense_centroids[%s]=%s, self.dense_sums[%s]=%s',
                fn, is_dense, dim,
                str(feat_mat.shape), idx, str(self.dense_centroids[idx].shape), idx,
                str(self.dense_sums[idx].shape))

        for idx, (fn, is_dense, dim, feat_mat, _, _) in enumerate(self.sparse_features):
            logging.debug('Initialize feature - name=%s, dense=%s, dim=%s, mat=%s', fn, is_dense, dim,
                         str(feat_mat.shape))
            self.sparse_feature_id[fn] = idx
            c, s = self.init_sparse_feature(dim)
            self.sparse_centroids[idx] = s
            self.sparse_sums[idx] = c

    def get_centroid_batch(self, i):
        pass

    def get_centroid(self, i):
        return np.expand_dims(self.centroids[i], 0)

    # def compute_centroid_multi_feature(self, i):
    #     for idx in range(len(self.dense_features)):
    #         # if the node is new we don't need to zero
    #         if not self.new_node[i]:
    #             self.dense_centroids[idx][i] *= 0
    #         self.dense_centroids[idx][i] += self.dense_sums[idx][i]
    #         self.dense_centroids[idx][i] /= self.num_descendants[i]
    #         norm = np.linalg.norm(self.dense_centroids[idx][i]) if self.norm == 'l2' else 0.0
    #         if norm != 0.0:
    #             self.dense_centroids[idx][i] /= norm
    #         logging.log_first_n(logging.DEBUG, 'self.dense_centroids[%s][%s].shape=%s, %s', 10, idx, i,
    #                             str(self.dense_centroids[idx][i].shape), str(self.dense_centroids[idx].shape))
    #     for idx in range(len(self.sparse_features)):
    #         self.sparse_centroids[idx][i] = self.sparse_sums[idx][i].copy()
    #         self.sparse_centroids[idx][i] /= self.num_descendants[i]
    #         norm = (self.sparse_centroids[idx][i] @ self.sparse_centroids[idx][i].T).todense().A[0,0] if self.norm == 'l2' else 0.0
    #         if norm != 0.0:
    #             self.sparse_centroids[idx][i] /= norm
    #         logging.log_first_n(logging.DEBUG, 'self.sparse_centroids[%s][%s].shape=%s', 10, idx, i,
    #                             str(self.sparse_centroids[idx][i].shape))

    def compute_centroid_multi_feature(self, i, update_sums=False):
        for idx in range(len(self.dense_features)):
            if update_sums:
                self.dense_sums[idx][i] *= 0
                self.dense_sums[idx][i] += self.dense_sums[idx][self.children[i]].sum(axis=0)
            if self.dense_features[idx][5] == CentroidType.NORMED:
                self.dense_compute_centroid_l2_norm(idx, i)
            elif self.dense_features[idx][5] == CentroidType.BINARY:
                self.dense_compute_centroid_binary(idx, i)
            elif self.dense_features[idx][5] == CentroidType.NO_NORM:
                self.dense_compute_centroid_no_norm(idx, i)
            logging.log_first_n(logging.DEBUG, 'self.dense_centroids[%s][%s].shape=%s, %s', 10, idx, i,
                                str(self.dense_centroids[idx][i].shape), str(self.dense_centroids[idx].shape))
        for idx in range(len(self.sparse_features)):
            if update_sums:
                assert len(self.children[i]) == 2
                self.sparse_sums[idx][i] = self.sparse_sums[idx][self.children[i][0]] + self.sparse_sums[idx][
                    self.children[i][1]]
            if self.sparse_features[idx][5] == CentroidType.NORMED:
                self.sparse_compute_centroid_l2_norm(idx, i)
            elif self.sparse_features[idx][5] == CentroidType.BINARY:
                self.sparse_compute_centroid_binary(idx, i)
            elif self.sparse_features[idx][5] == CentroidType.NO_NORM:
                self.sparse_compute_centroid_no_norm(idx, i)
            logging.log_first_n(logging.DEBUG, 'self.sparse_centroids[%s][%s].shape=%s', 10, idx, i,
                                str(self.sparse_centroids[idx][i].shape))

    def dense_compute_centroid_l2_norm(self, idx, i):
        self.dense_compute_centroid_no_norm(idx, i)
        norm = np.linalg.norm(self.dense_centroids[idx][i]) if self.norm == 'l2' else 0.0
        if norm != 0.0:
            self.dense_centroids[idx][i] /= norm

    def dense_compute_centroid_binary(self, idx, i):
        self.dense_compute_centroid_no_norm(idx, i)
        self.dense_centroids[idx][i] = (self.dense_centroids[idx][i] > 0).astype(np.float32)

    def dense_compute_centroid_no_norm(self, idx, i):
        self.dense_centroids[idx][i] *= 0
        self.dense_centroids[idx][i] += self.dense_sums[idx][i]
        self.dense_centroids[idx][i] /= self.num_descendants[i]

    def sparse_compute_centroid_l2_norm(self, idx, i):
        self.sparse_compute_centroid_no_norm(idx, i)
        norm = (self.sparse_centroids[idx][i] @ self.sparse_centroids[idx][i].T).todense().A[
            0, 0]
        if norm != 0.0:
            self.sparse_centroids[idx][i] /= norm

    def sparse_compute_centroid_binary(self, idx, i):
        self.sparse_compute_centroid_no_norm(idx, i)
        self.sparse_centroids[idx][i].data = (self.sparse_centroids[idx][i].data > 0).astype(np.float32)

    def sparse_compute_centroid_no_norm(self, idx, i):
        self.sparse_centroids[idx][i] = self.sparse_sums[idx][i].copy()

    def get_sum(self, i):
        return np.expand_dims(self.sums[i], 0)

    def set_sum(self, i, v):
        self.sums[i] = v

    def single_update(self, i):
        logging.debug('updating node %s', i)
        assert self.needs_update_model[i]
        kids = self.get_children(i)
        s = time.time()
        assert len(kids) == 2, 'node %s has kids %s (needed 2 children!)' % (i, str(kids))
        self.num_descendants[i] = self.num_descendants[kids[0]] + self.num_descendants[kids[1]]
        self.compute_centroid_multi_feature(i, update_sums=True)
        self.time_in_centroid += time.time() - s
        self.new_node[i] = False
        self.needs_update_model[i] = False

    def insert(self, i):
        if self.points_set is False:
            logging.debug('grinch points not set, setting now for all points.....')
            self.set_points(np.arange(self.num_points))
        s = time.time()
        logging.debug('[insert] insert(%s) pc=%s', i, self.point_counter)
        # first point
        if self.point_counter == 0:
            self.add_pt(i)
        else:
            self.add_pt(i)
            dists, nns = self.cknn(np.array([i], dtype=np.int32), self.k, None, None, pc=self.point_counter - 1)
            logging.debug('[insert] dists=%s nns=%s', str(dists), str(nns))
            sib = self.find_rotate(i, nns[0])
            logging.debug('[insert] sib=%s', sib)
            parent = self.node_from_nodes(sib, i)
            self.make_sibling(sib, i, parent)
            curr_update = parent
            while curr_update != -1:
                self.updated_from_children(curr_update)
                curr_update = self.get_parent(curr_update)
            if self.perform_graft:
                self.graft(parent)
        t = time.time()
        logging.debug('[insert] finished insert(%s) in %s seconds', i, t - s)

    def graft(self, gnode):
        s = time.time()
        logging.debug('[graft] graft(%s)', gnode)
        curr = gnode

        # Find offlimits
        offlimits1 = self.get_descendants(curr)
        if self.get_parent(curr) != -1 and len(self.get_children(self.get_sibling(curr))) == 0:
            offlimits2 = [self.get_sibling(curr)]
        else:
            offlimits2 = []

        logging.debug('[graft] len(offlimits1)=%s len(offlimits2)=%s', len(offlimits1), len(offlimits2))
        logging.debug('[graft] offlimits1 %s offlimits2 %s', str(offlimits1), str(offlimits2))

        # Find updates
        self.update(curr)
        curr_v = np.array([curr], dtype=np.int32)

        # Do search
        search_st = time.time()
        _, nns = self.cknn(curr_v, self.k, offlimits1, offlimits2)
        self.time_in_graft_search += time.time() - search_st
        self.this_time_in_graft_search += time.time() - search_st
        if len(nns) == 0:
            logging.debug('No nearest neighbors after nns....')
            return

        oneNN = nns[0]
        logging.debug('Nearest neighbor is %s', oneNN)
        lca, this2anc, other2anc = self.lca_and_ancestors(gnode, oneNN)
        logging.debug('lca %s len(this2anc) %s len(other2anc) %s', lca, len(this2anc), len(other2anc))
        if this2anc and other2anc:
            # find all pairwise distances
            # this_vecs = self.get_centroid(this2anc)
            # anc_vecs = self.get_centroid(other2anc)

            # M by N
            M = len(this2anc)
            N = len(other2anc)
            score_if_grafted = self.e_score_batch(this2anc, other2anc)
            assert score_if_grafted.shape[0] == M
            assert score_if_grafted.shape[1] == N
            # 1 by N
            nn_parent_score = np.expand_dims(self.get_score_batch(self.get_parent(other2anc)), 0)
            assert nn_parent_score.shape[0] == 1
            assert nn_parent_score.shape[1] == N
            # M by 1
            curr_parent_score = np.expand_dims(self.get_score_batch(self.get_parent(this2anc)), 1)
            assert curr_parent_score.shape[0] == M
            assert curr_parent_score.shape[1] == 1

            not_i_like_you = score_if_grafted <= curr_parent_score
            not_you_like_me = score_if_grafted <= nn_parent_score
            assert not_i_like_you.shape[0] == M
            assert not_i_like_you.shape[1] == N
            assert not_you_like_me.shape[0] == M
            assert not_you_like_me.shape[1] == N

            graft_condition = not_i_like_you | not_you_like_me
            num_meeting_condition = graft_condition.sum()
            self.number_of_graft_allowable += num_meeting_condition

            score_if_grafted[graft_condition] = 0
            argmax = np.argmax(score_if_grafted)
            argmax_row = int(argmax / score_if_grafted.shape[1])
            argmax_col = argmax % score_if_grafted.shape[1]
            best_1 = this2anc[argmax_row]
            best_2 = other2anc[argmax_col]
            if not not_i_like_you[argmax_row, argmax_col] and not not_you_like_me[argmax_row, argmax_col]:
                self.number_of_grafts += 1
                self.this_number_of_grafts += 1
                bestPair2gp = self.get_parent(self.get_parent(best_2))
                parent = self.node_from_nodes(best_1, best_2)
                self.make_sibling(best_1, best_2, parent)
                logging.debug('[graft] node %s grafts node %s, scores %s > max(%s, %s)' % (best_1, best_2,
                                                                                           score_if_grafted[argmax_row,
                                                                                                            argmax_col],
                                                                                           curr_parent_score[
                                                                                               argmax_row, 0],
                                                                                           nn_parent_score[
                                                                                               0, argmax_col]))
                for start in [bestPair2gp, self.get_parent(curr)]:
                    curr_update = start
                    while curr_update != -1:
                        self.updated_from_children(curr_update)
                        curr_update = self.get_parent(curr_update)
            else:
                logging.debug('[graft] There was no case where we wanted to graft.')
        self.number_of_grafts_considered += len(this2anc) * len(other2anc)
        self.time_in_graft += time.time() - s

        self.this_number_of_grafts_considered = len(this2anc) * len(other2anc)
        self.this_time_in_graft += time.time() - s

    def cknn(self, i_vec, k, offlimits1, offlimits2, pc=None):
        k = min(self.point_counter, k)
        s = time.time()
        # TODO: Allow offlimits to 2D
        # import pdb; pdb.set_trace()
        if pc is None:
            pc = self.point_counter
        logging.debug('[cknn] i_vec %s point_vecs %s', str(i_vec), str(pc))
        sims = self.csim_multi_feature_knn(i_vec, np.arange(0, pc))
        logging.debug('[cknn] sims.shape %s point_vecs', str(sims.shape))
        if offlimits1 is not None:
            sims[:, offlimits1] = -float("Inf")
        if offlimits2 is not None:
            sims[:, offlimits2] = -float("Inf")
        indices = np.argmax(sims, axis=1)
        distances = sims[0, indices]
        indices = indices[distances != -np.Inf]
        distances = distances[distances != -np.Inf]
        self.time_in_search += time.time() - s
        # print(distances)
        return distances, indices

    def e_score_batch(self, i, j):
        if np.any(self.needs_update_model[i]):
            for ii in i:
                if self.needs_update_model[ii]:
                    self.update(ii)
        if np.any(self.needs_update_model[j]):
            for jj in j:
                if self.needs_update_model[jj]:
                    self.update(jj)

        # self.get_centroid_batch(i)
        # self.get_centroid_batch(j)

        s1 = time.time()
        # sims = np.matmul(i_vec, j_vec.transpose(1, 0))
        sims = self.csim(i, j)
        self.time_in_graft_score_only += time.time() - s1
        return sims

    def csim_multi_feature(self, i, j):
        """

        :param i:
        :param j:
        :return:
        """
        len_i = len(i)
        len_j = len(j)
        s = np.zeros((len_i, len_j), dtype=np.float32)
        for idx in range(len(self.dense_centroids)):
            t = time.time()
            feat_score = self.dense_centroids[idx][i] @ self.dense_centroids[idx][j].T
            s += feat_score
            total_time = time.time() - t
            # wandb.log({'time/%s_csim' % self.dense_features[idx][0]: total_time, 'point_counter': self.point_counter})
        for idx in range(len(self.sparse_centroids)):
            # import pdb
            t = time.time()
            # pdb.set_trace()
            feat_score = (scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) @ scipy.sparse.vstack(
                [self.sparse_centroids[idx][jj] for jj in j]).T).todense().A
            # feat_score = np.zeros((len_i, len_j), dtype=np.float32)
            # for idxii, ii in enumerate(i):
            #     for idxjj, jj in enumerate(j):
            #         feat_score[idxii, idxjj] = self.sparse_centroids[idx][ii].dot(self.sparse_centroids[idx][jj].T).todense().A
            # feat_score = (self.sparse_centroids[idx][i] @ self.sparse_centroids[idx][j].T).todense().A
            s += feat_score
            total_time = time.time() - t
            # wandb.log({'time/%s_csim' % self.sparse_features[idx][0]: total_time, 'point_counter': self.point_counter})
        return s

    def csim_multi_feature_knn(self, i, j):
        """

        :param i:
        :param j:
        :return:
        """
        len_i = len(i)
        len_j = len(j)
        s = np.zeros((len_i, len_j), dtype=np.float32)
        for idx in range(len(self.dense_centroids)):
            t = time.time()
            feat_score = self.dense_centroids[idx][i] @ self.dense_point_features[idx][j].T
            s += feat_score
            total_time = time.time() - t
            # wandb.log({'time/%s_csim' % self.dense_features[idx][0]: total_time, 'point_counter': self.point_counter})
        for idx in range(len(self.sparse_centroids)):
            # import pdb
            t = time.time()
            # pdb.set_trace()
            lhs = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 0 else \
                self.sparse_centroids[idx][i[0]]
            rhs = self.sparse_point_features[idx][j].T
            feat_score = (lhs @ rhs).todense().A
            # feat_score = (self.sparse_centroids[idx][i] @ self.sparse_centroids[idx][j].T).todense().A
            s += feat_score
            total_time = time.time() - t
            # wandb.log({'time/%s_csim' % self.sparse_features[idx][0]: total_time, 'point_counter': self.point_counter})
        return s

    def sim(self, i, j):
        # i[0] dense, i[1] sparse
        # i[0][0].shape[0] is num examples
        # len_i = i[0][0].shape[0] if len(i[0]) > 0 else i[0][1].shape[0]
        # len_j = j[0][0].shape[0] if len(j[0]) > 0 else j[0][1].shape[0]
        # import pdb; pdb.set_trace()
        s = 0
        for idx in range(len(self.dense_centroids)):
            logging.log_first_n(logging.DEBUG, 'dense[%s] - %s @ %s.T', 10, idx,
                                str(self.dense_centroids[idx][None, i].shape),
                                str(self.dense_centroids[idx][None, j].shape))
            feat_score = self.dense_centroids[idx][None, i] @ self.dense_centroids[idx][None, j].T
            s += feat_score
        for idx in range(len(self.sparse_centroids)):
            logging.log_first_n(logging.DEBUG, 'sparse[%s] - %s @ %s.T', 10, idx,
                                str(self.sparse_centroids[idx][i].shape), str(self.sparse_centroids[idx][j].shape))
            feat_score = (self.sparse_centroids[idx][i] @ self.sparse_centroids[idx][j].T).todense().A
            s += feat_score
        return s

    def e_score(self, i, j):
        if self.needs_update_model[i]:
            self.update(i)

        if self.needs_update_model[j]:
            self.update(j)

        sims = self.sim(i, j)
        return sims

    def get_score_batch(self, i):
        s = time.time()
        if not np.all(np.isfinite(self.scores[i])):
            to_set = np.array([ii for ii in i if not np.isfinite(self.scores[ii])], dtype=np.int32)
            needed_i = np.array([self.children[ii][0] for ii in to_set], dtype=np.int32)
            needed_j = np.array([self.children[ii][1] for ii in to_set], dtype=np.int32)
            if np.any(self.needs_update_model[needed_i]):
                for ii in needed_i:
                    if self.needs_update_model[ii]:
                        self.update(ii)
            if np.any(self.needs_update_model[needed_j]):
                for jj in needed_j:
                    if self.needs_update_model[jj]:
                        self.update(jj)
            self.scores[to_set] = self.pw_sim(needed_i, needed_j)
        self.time_in_graft_get_scores += time.time() - s
        self.this_time_in_graft_get_scores += time.time() - s
        return self.scores[i]

    def all_thresholds(self):
        self.update(self.root(), use_tqdm=True)
        return self.get_score_batch(self.all_valid_internal_nodes())

    def pw_sim(self, i, j, record_dict=None):
        assert len(i) == len(j)
        if np.any(self.needs_update_model[i]):
            for ii in i:
                if self.needs_update_model[ii]:
                    self.update(ii)
        if np.any(self.needs_update_model[j]):
            for jj in j:
                if self.needs_update_model[jj]:
                    self.update(jj)
        s = np.zeros(len(i), dtype=np.float32)
        # for idx in range(len(self.dense_centroids)):
        #     logging.log_first_n(logging.DEBUG, 'dense[%s] - %s @ %s.T', 10, idx, str(self.dense_centroids[idx][i].shape), str(self.dense_centroids[idx][j].shape))
        #     if self.dense_features[idx][4] == FeatCalc.L2:
        #         feat_score = w * self.p_l2dist_feature_dense(idx, i, j)
        #     elif self.dense_features[idx][4] == FeatCalc.DOT:
        #         feat_score = w * self.p_dot_feature_dense(idx, i, j)
        #     elif self.dense_features[idx][4] == FeatCalc.L2_gt_one:
        #         feat_score = w * self.p_l2dist_gt_one_feature_dense(idx, i, j)
        #     if b is not None:
        #         feat_score += b
        #     if record_dict is not None:
        #         record_dict[self.dense_features[idx][0]] = feat_score
        #     s += feat_score
        idx_offset = len(self.dense_features)
        for idx in range(len(self.sparse_centroids)):
            if self.sparse_features[idx][4] == FeatCalc.DOT:
                c_i = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 1 else \
                    self.sparse_centroids[idx][i[0]]
                c_j = scipy.sparse.vstack([self.sparse_centroids[idx][jj] for jj in j]) if len(j) > 1 else \
                    self.sparse_centroids[idx][j[0]]
                res = c_i.multiply(c_j).sum(axis=1).A.astype(np.float32)
                feat_score = res[:, 0]
                s += feat_score
        return s


class WeightedMultiFeatureGrinch(MultiFeatureGrinch):

    def __init__(self, model, features, num_points, dim=None, rotate_cap=100, graft_cap=100, norm='none', sim='dot',
                 max_num_points=None, min_allowable_sim=-20000.0, pids=None):
        super(WeightedMultiFeatureGrinch, self).__init__(features, num_points,
                                                         dim=dim, rotate_cap=rotate_cap,
                                                         graft_cap=graft_cap, norm=norm, sim=sim, max_num_points=max_num_points)
        self.model = model
        logging.debug('Using len(features)=%s', len(features))
        self.min_allowable_sim = min_allowable_sim
        self.pids = pids

    @staticmethod
    def from_agglom(agglom, pids=None, canopies=None):
        # Set tree structure
        grinch = WeightedMultiFeatureGrinch(agglom.model, agglom.features, agglom.num_points)
        grinch.from_scipy_z(agglom.Z,pids=pids, canopies=canopies)
        grinch.min_allowable_sim = agglom.min_allowable_sim
        return grinch

    def similarity_threshold_from_agglom(self, threshold):
        """Transform distance threshold to similarity one"""
        pos_sim = threshold - np.minimum(0.0, self.min_allowable_sim)

    def save_and_quit(self, filename):
        # remove features
        # unset leaf features
        self.prepare_for_save()
        # save tree structure
        with open(filename, 'wb') as fout:
            pickle.dump(self, fout)

    def prepare_for_save(self):
        self.clear_node_features()
        for i in self.all_valid_internal_nodes():
            self.descendants[i] = []
            self.needs_update_desc[i] = True
            self.needs_update_model[i] = True
        self.points_set = False

    def build_dendrogram_hac(self):
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
        st_add = time.time()
        # for i in range(self.num_points):
        #     self.add_pt(i)
        self.set_points([x for x in range(0, self.num_points)])
        en_add = time.time()
        logging.debug('Time to add points: %s', en_add - st_add)
        st_sims = time.time()
        sims = self.csim_multi_feature_knn_batched(np.arange(self.num_points), np.arange(self.num_points))
        en_sims = time.time()
        logging.debug('Time to compute sims: %s', en_sims - st_sims)
        logging.debug('Finished batched similarities!')
        st_prep = time.time()
        dists = sims.max() - sims
        dists = (dists + dists.T) / 2.0
        np.fill_diagonal(dists, 0.0)
        dists = squareform(dists)
        en_prep = time.time()
        logging.debug('Time to compute sims: %s', en_prep - st_prep)
        logging.debug('Finished preparing distances!')
        logging.debug('Running hac')
        st_linkage = time.time()
        Z = linkage(dists, method='average')
        en_linkage = time.time()
        logging.debug('Time to run HAC: %s', en_linkage - st_linkage)
        logging.debug('Finished hac!')
        st_z = time.time()
        self.from_scipy_z(Z)
        en_z = time.time()
        logging.debug('from_scipy_z: %s', en_z - st_z)

    def csim_multi_feature(self, i, j):
        """

        :param i:
        :param j:
        :return:
        """
        return self.csim_multi_feature_torch(i, j).detach().numpy()

    def csim_multi_feature_torch(self, i, j):
        """

        :param i:
        :param j:
        :return:
        """
        len_i = len(i)
        len_j = len(j)
        s = torch.zeros((len_i, len_j), dtype=torch.float32)
        for idx in range(len(self.dense_centroids)):
            w, b = self.model.weight_for(self.dense_features[idx][0])
            if self.dense_features[idx][4] == FeatCalc.L2:
                feat_score = w * self.c_l2dist_feature_dense(idx, i, j)
            elif self.dense_features[idx][4] == FeatCalc.DOT:
                feat_score = w * self.c_dot_feature_dense(idx, i, j)
            elif self.dense_features[idx][4] == FeatCalc.L2_gt_one:
                feat_score = w * self.c_l2dist_gt_one_feature_dense(idx, i, j)
            elif self.dense_features[idx][4] == FeatCalc.NO_MATCH:
                feat_score = w * self.c_no_match_feature_dense(idx, i, j)
            if b is not None:
                feat_score += b
            s += feat_score
        for idx in range(len(self.sparse_centroids)):
            w, b = self.model.weight_for(self.sparse_features[idx][0])
            if self.sparse_features[idx][4] == FeatCalc.L2:
                feat_score = w * self.c_l2dist_feature_sparse(idx, i, j)
            elif self.sparse_features[idx][4] == FeatCalc.DOT:
                feat_score = w * self.c_dot_feature_sparse(idx, i, j)
            elif self.sparse_features[idx][4] == FeatCalc.L2_gt_one:
                feat_score = w * self.c_l2dist_gt_one_feature_dense(idx, i, j)
            if b is not None:
                feat_score += b
            s += feat_score
        return s

    def csim_multi_feature_knn_batched(self, i, j, batch_size=1000):
        len_i = len(i)
        len_j = len(j)
        should_log = len_i > 3 * batch_size
        if should_log:
            logging.debug('[csim_multi_feature_knn_batched] %s by %s', len_i, len_j)
        res = np.zeros((len_i, len_j), dtype=np.float32)
        i_batches = [x for x in range(0, len_i, batch_size)]
        st = time.time()
        for idx, ii in enumerate(i_batches):
            if should_log:
                logging.debug('[csim_multi_feature_knn_batched] %s by %s - %s of %s in % seconds', len_i, len_j, idx,
                             len(i_batches) - 1, time.time() - st)
            for jj in range(0, len_j, batch_size):
                istart = ii
                jstart = jj
                iend = min(len_i, ii + batch_size)
                jend = min(len_j, jj + batch_size)
                res[istart:iend, jstart:jend] = self.csim_multi_feature_knn(i[istart:iend], j[jstart:jend]).astype(
                    np.float32)
        logging.debug('[csim_multi_feature_knn_batched] DONE! %s by %s - %s of %s in % seconds', len_i, len_j,
                     len(i_batches) - 1,
                     len(i_batches) - 1, time.time() - st)
        return res

    def csim_multi_feature_knn(self, i, j):
        """

        :param i:
        :param j:
        :return:
        """
        return self.csim_multi_feature_knn_torch(i, j).detach().numpy()

    def csim_multi_feature_knn_torch(self, i, j, record_dict=None):
        len_i = len(i)
        len_j = len(j)
        s = torch.zeros((len_i, len_j), dtype=torch.float32)
        logging.log_first_n(logging.DEBUG, 'csim_multi_feature_knn_torch(i=%s, j=%s)', 1, str(i), str(j))
        logging.log_first_n(logging.DEBUG, 'len(self.dense_features)=%s', 1, len(self.dense_features))
        logging.log_first_n(logging.DEBUG, 'len(self.dense_sums)=%s', 1, len(self.dense_sums))
        logging.log_first_n(logging.DEBUG, 'len(self.dense_centroids)=%s', 1, len(self.dense_centroids))

        logging.log_first_n(logging.DEBUG, 'len(self.sparse_features)=%s', 1, len(self.sparse_features))
        logging.log_first_n(logging.DEBUG, 'len(self.sparse_sums)=%s', 1, len(self.sparse_sums))
        logging.log_first_n(logging.DEBUG, 'len(self.sparse_centroids)=%s', 1, len(self.sparse_centroids))

        for idx in range(len(self.dense_features)):
            logging.log_first_n(logging.DEBUG, 'idx=%s', 10, idx)
            logging.log_first_n(logging.DEBUG, 'self.dense_features[idx][0]=%s', 10, self.dense_features[idx][0])
            w, b = self.model.weight_for(self.dense_features[idx][0])
            rhs = self.dense_point_features[idx][j]  # Grab the original feature matrix for j
            if self.dense_features[idx][4] == FeatCalc.L2:
                feat_score = w * self.c_l2dist_feature_dense_knn(idx, i, rhs)
            elif self.dense_features[idx][4] == FeatCalc.DOT:
                feat_score = w * self.c_dot_feature_dense_knn(idx, i, rhs)
            elif self.dense_features[idx][4] == FeatCalc.L2_gt_one:
                feat_score = w * self.c_l2dist_gt_one_feature_dense_knn(idx, i, rhs)
            elif self.dense_features[idx][4] == FeatCalc.NO_MATCH:
                feat_score = w * self.c_no_match_feature_dense_knn(idx, i, rhs)
            if b is not None:
                feat_score += b
            if record_dict is not None:
                record_dict[self.dense_features[idx][0]] = feat_score
            s += feat_score
        for idx in range(len(self.sparse_features)):
            w, b = self.model.weight_for(self.sparse_features[idx][0])
            rhs = self.sparse_point_features[idx][j]  # Grab the original feature matrix for j
            if self.sparse_features[idx][4] == FeatCalc.L2:
                feat_score = w * self.c_l2dist_feature_sparse_knn(idx, i, rhs)
            elif self.sparse_features[idx][4] == FeatCalc.DOT:
                feat_score = w * self.c_dot_feature_sparse_knn(idx, i, rhs)
            elif self.sparse_features[idx][4] == FeatCalc.L2_gt_one:
                feat_score = w * self.c_l2dist_gt_one_feature_sparse_knn(idx, i, rhs)
            if b is not None:
                feat_score += b
            if record_dict is not None:
                record_dict[self.sparse_features[idx][0]] = feat_score
            s += feat_score
        return s

    def sim_torch(self, i, j):
        return self.pw_sim_torch([i], [j])[0]

    def pw_sim_torch(self, i, j, record_dict=None):
        assert len(i) == len(j)
        s = torch.zeros(len(i), dtype=torch.float32)
        for idx in range(len(self.dense_features)):
            w, b = self.model.weight_for(self.dense_features[idx][0])
            logging.log_first_n(logging.DEBUG, 'dense[%s] - %s @ %s.T', 10, idx, str(self.dense_centroids[idx][i].shape),
                                str(self.dense_centroids[idx][j].shape))
            if self.dense_features[idx][4] == FeatCalc.L2:
                feat_score = w * self.p_l2dist_feature_dense(idx, i, j)
            elif self.dense_features[idx][4] == FeatCalc.DOT:
                feat_score = w * self.p_dot_feature_dense(idx, i, j)
            elif self.dense_features[idx][4] == FeatCalc.L2_gt_one:
                feat_score = w * self.p_l2dist_gt_one_feature_dense(idx, i, j)
            elif self.dense_features[idx][4] == FeatCalc.NO_MATCH:
                feat_score = w * self.p_no_match_feature_dense(idx, i, j)
            if b is not None:
                feat_score += b
            if record_dict is not None:
                record_dict[self.dense_features[idx][0]] = feat_score
            s += feat_score
        idx_offset = len(self.dense_features)
        for idx in range(len(self.sparse_features)):
            w, b = self.model.weight_for(self.sparse_features[idx][0])
            if self.sparse_features[idx][4] == FeatCalc.L2:
                feat_score = w * self.p_l2dist_feature_sparse(idx, i, j)
            elif self.sparse_features[idx][4] == FeatCalc.DOT:
                feat_score = w * self.p_dot_feature_sparse(idx, i, j)[:, 0]
            elif self.sparse_features[idx][4] == FeatCalc.L2_gt_one:
                feat_score = w * self.p_l2dist_gt_one_feature_sparse(idx, i, j)
            if b is not None:
                feat_score += b
            if record_dict is not None:
                record_dict[self.sparse_features[idx][0]] = feat_score
            s += feat_score
        return s

    def sim(self, i, j):
        return self.sim_torch(i, j).detach().numpy()

    def pw_sim(self, i, j):
        return self.pw_sim_torch(i, j).detach().numpy()

    def p_l2dist_feature_dense(self, idx, i, j):
        c_i = self.dense_centroids[idx][i]
        c_j = self.dense_centroids[idx][j]
        res = paired_distances(c_i, c_j, metric='euclidean') ** 2
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_feature_dense(self, idx, i, j):
        c_i = self.dense_centroids[idx][i]
        c_j = self.dense_centroids[idx][j]
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_feature_dense_knn(self, idx, i, c_j):
        c_i = self.dense_centroids[idx][i]
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2
        return torch.from_numpy(res.astype(np.float32))

    def p_l2dist_gt_one_feature_dense(self, idx, i, j):
        c_i = self.dense_centroids[idx][i]
        c_j = self.dense_centroids[idx][j]
        res = (paired_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2) > 1
        return torch.from_numpy(res.astype(np.float32))

    def p_no_match_feature_dense(self, idx, i, j):
        c_i = self.dense_centroids[idx][i]
        c_j = self.dense_centroids[idx][j]
        res = np.squeeze(np.logical_and(np.logical_and(c_i != c_j, c_i != -1), c_j != -1), 1)
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_gt_one_feature_dense(self, idx, i, j):
        c_i = self.dense_centroids[idx][i]
        c_j = self.dense_centroids[idx][j]
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2 > 1
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_gt_one_feature_dense_knn(self, idx, i, c_j):
        c_i = self.dense_centroids[idx][i]
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2 > 1
        return torch.from_numpy(res.astype(np.float32))

    def c_no_match_feature_dense(self, idx, i, j):
        c_i = self.dense_centroids[idx][i]
        c_jT = self.dense_centroids[idx][j].T
        res = np.logical_and(np.logical_and(c_i != c_jT, c_i != -1), c_jT != -1)
        return torch.from_numpy(res.astype(np.float32))

    def c_no_match_feature_dense_knn(self, idx, i, c_j):
        c_i = self.dense_centroids[idx][i]
        c_jT = c_j.T
        res = np.logical_and(np.logical_and(c_i != c_jT, c_i != -1), c_jT != -1)
        return torch.from_numpy(res.astype(np.float32))

    def p_dot_feature_dense(self, idx, i, j):
        c_i = self.dense_centroids[idx][i]
        c_j = self.dense_centroids[idx][j]
        res = np.sum(c_i * c_j, axis=1).astype(np.float32)
        return torch.from_numpy(res.astype(np.float32))

    def c_dot_feature_dense(self, idx, i, j):
        c_i = self.dense_centroids[idx][i]
        c_j = self.dense_centroids[idx][j]
        res = np.matmul(c_i, c_j.T)
        return torch.from_numpy(res.astype(np.float32))

    def c_dot_feature_dense_knn(self, idx, i, c_j):
        c_i = self.dense_centroids[idx][i]
        res = np.matmul(c_i, c_j.T)
        return torch.from_numpy(res.astype(np.float32))

    def p_l2dist_feature_sparse(self, idx, i, j):
        c_i = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 1 else \
            self.sparse_centroids[idx][i[0]]
        c_j = scipy.sparse.vstack([self.sparse_centroids[idx][jj] for jj in j]) if len(j) > 1 else \
            self.sparse_centroids[idx][j[0]]
        res = paired_distances(c_i, c_j, metric='euclidean') ** 2
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_feature_sparse(self, idx, i, j):
        c_i = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 1 else \
            self.sparse_centroids[idx][i[0]]
        c_j = scipy.sparse.vstack([self.sparse_centroids[idx][jj] for jj in j]) if len(j) > 1 else \
            self.sparse_centroids[idx][j[0]]
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_feature_sparse_knn(self, idx, i, c_j):
        c_i = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 1 else \
            self.sparse_centroids[idx][i[0]]
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2
        return torch.from_numpy(res.astype(np.float32))

    def p_l2dist_gt_one_feature_sparse(self, idx, i, j):
        c_i = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 1 else \
            self.sparse_centroids[idx][i[0]]
        c_j = scipy.sparse.vstack([self.sparse_centroids[idx][jj] for jj in j]) if len(j) > 1 else \
            self.sparse_centroids[idx][j[0]]
        res = paired_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2 > 1
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_gt_one_feature_sparse(self, idx, i, j):
        c_i = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 1 else \
            self.sparse_centroids[idx][i[0]]
        c_j = scipy.sparse.vstack([self.sparse_centroids[idx][jj] for jj in j]) if len(j) > 1 else \
            self.sparse_centroids[idx][j[0]]
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2 > 1
        return torch.from_numpy(res.astype(np.float32))

    def c_l2dist_gt_one_feature_sparse_knn(self, idx, i, c_j):
        c_i = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 1 else \
            self.sparse_centroids[idx][i[0]]
        res = pairwise_distances(c_i, c_j, metric='euclidean', n_jobs=-1) ** 2 > 1
        return torch.from_numpy(res.astype(np.float32))

    def p_dot_feature_sparse(self, idx, i, j):
        c_i = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 1 else \
            self.sparse_centroids[idx][i[0]]
        c_j = scipy.sparse.vstack([self.sparse_centroids[idx][jj] for jj in j]) if len(j) > 1 else \
            self.sparse_centroids[idx][j[0]]
        res = c_i.multiply(c_j).sum(axis=1).A.astype(np.float32)
        return torch.from_numpy(res.astype(np.float32))

    def c_dot_feature_sparse(self, idx, i, j):
        c_i = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 1 else \
            self.sparse_centroids[idx][i[0]]
        c_j = scipy.sparse.vstack([self.sparse_centroids[idx][jj] for jj in j])
        res = (c_i @ c_j.T).todense().A
        return torch.from_numpy(res.astype(np.float32))

    def c_dot_feature_sparse_knn(self, idx, i, c_j):
        c_i = scipy.sparse.vstack([self.sparse_centroids[idx][ii] for ii in i]) if len(i) > 1 else \
            self.sparse_centroids[idx][i[0]]
        res = (c_i @ c_j.T).todense().A
        return torch.from_numpy(res.astype(np.float32))
