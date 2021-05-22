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

import pickle
import time
from string import punctuation

import numpy as np

try:
    import sent2vec
except:
    print("no sent2vec")

from absl import logging
from nltk import word_tokenize
from nltk.corpus import stopwords
import scipy.sparse

logging.set_verbosity(logging.INFO)

must_not_link_rule = -20000.0
must_link_rule = 5000.0

from enum import Enum


class FeatCalc(Enum):
    L2 = 1
    DOT = 2
    L2_gt_one = 3
    NO_MATCH = 4
    LOCATION = 5
    STRING = 6


class CentroidType(Enum):
    BINARY = 1
    NORMED = 2
    NO_NORM = 3


class FastTextFeatures(object):
    """Features using a fasttext / sent2vec model."""

    def __init__(self, filename, name, get_field):
        self.filename = filename
        self.name = name
        self.get_field = get_field
        logging.info('Loading model from %s...' % filename)
        t = time.time()
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(self.filename)
        logging.info('Finished loading %s.', time.time() - t)
        self.stop_words = set(stopwords.words('english'))

    def encode(self, things_to_encode):
        res = np.vstack(
            [self.model.embed_sentence(self.preprocess_sentence(self.get_field(x))) for x in things_to_encode]).astype(
            np.float32)
        logging.log_first_n(logging.INFO, 'len(things_to_encode) = %s, res.shape %s', 10, len(things_to_encode),
                            str(res.shape))
        norms = np.linalg.norm(res, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        res /= norms
        logging.log_first_n(logging.INFO, 'len(things_to_encode) = %s, res.shape %s', 10, len(things_to_encode),
                            str(res.shape))
        return res

    def preprocess_sentence(self, text):
        text = text.replace('/', ' / ')
        text = text.replace('.-', ' .- ')
        text = text.replace('.', ' . ')
        text = text.replace('\'', ' \' ')
        text = text.lower()

        tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in self.stop_words]

        return ' '.join(tokens)

    def __getstate__(self):
        self.model = None


class SKLearnVectorizerFeatures(object):
    """Features for SKLearn vectorizer."""

    def __init__(self, filename, name, get_field):
        self.filename = filename
        self.name = name
        self.get_field = get_field
        logging.info('Loading model from %s...' % filename)
        t = time.time()
        with open(self.filename, 'rb') as fin:
            self.model = pickle.load(fin)
        logging.info('Finished loading %s.', time.time() - t)

    def encode(self, things_to_encode):
        logging.log_first_n(logging.INFO, 'len(things_to_encode) = %s, %s', 10, len(things_to_encode),
                            ', '.join([str(self.get_field(x)) for x in things_to_encode[:5]]))
        return self.model.transform([self.get_field(x) for x in things_to_encode])


class HashingVectorizerFeatures(object):
    """Features for hashing vectorizer."""

    def __init__(self, name, get_field, norm=None):
        self.name = name
        self.get_field = get_field
        from sklearn.feature_extraction.text import HashingVectorizer
        self.model = HashingVectorizer(analyzer=lambda x: [xx for xx in x], alternate_sign=False, dtype=np.float32,
                                       norm=norm)

    def encode(self, things_to_encode):
        return self.model.transform([self.get_field(x) for x in things_to_encode])


class SingleItemHashingVectorizerFeatures(object):
    """Features for hashing vectorizer where each feature is a single integer."""

    def __init__(self, name, get_field, norm=None):
        self.name = name
        self.get_field = get_field
        from sklearn.feature_extraction.text import HashingVectorizer
        self.model = HashingVectorizer(analyzer=lambda x: [' '.join(x)] if x else [], alternate_sign=False,
                                       dtype=np.int32, norm=None)

    def encode(self, things_to_encode):
        res = self.model.transform([self.get_field(x) for x in things_to_encode])
        idx, val = res.nonzero()
        enc = -1 * np.ones(len(things_to_encode), dtype=np.int32)
        enc[idx] = val
        return np.expand_dims(enc, axis=1)


class EncodingModel(object):
    def __init__(self, feature_list, name, aux, feature_types, centroid_types, must_link_rules, must_not_link_rules):
        self.feature_list = feature_list
        self.name = name
        self.aux = aux
        self.feature_types = feature_types
        self.centroid_types = centroid_types
        self.must_link_rules = must_link_rules
        self.must_not_link_rules = must_not_link_rules
        logging.info('Creating model %s with %s features = [%s]',
                     name, len(feature_list), ', '.join([x.name for x in feature_list]))

    def encode(self, mentions):
        features = []
        for idx, f in enumerate(self.feature_list):
            # (fn, is_dense, num_points by dim, feat_mat)
            f_enc = f.encode(mentions)
            is_dense = not scipy.sparse.issparse(f_enc)
            features.append(
                (f.name, is_dense, f_enc.shape[1], f_enc, self.feature_types[idx], self.centroid_types[idx]))
        return features
