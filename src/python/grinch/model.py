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

import numpy as np
import torch.nn
from absl import logging

from grinch.features import must_not_link_rule, must_link_rule


class LinearAndRuleModel(torch.nn.Module):
    """A linear model w/ must and must-not link constraints."""

    def __init__(self, init_w, init_b, init_rules, feat2id):
        super(LinearAndRuleModel, self).__init__()
        self.lin_weight = torch.nn.Parameter(torch.from_numpy(init_w))
        self.bias = torch.nn.Parameter(torch.from_numpy(init_b))
        self.rules = torch.from_numpy(init_rules)
        self.feat2id = feat2id
        self.aux = dict()

    def weight_for(self, feature_name):
        learned, fid = self.feat2id[feature_name]
        if learned:
            return self.lin_weight[fid], self.bias[fid]
        else:
            return self.rules[fid], None

    @staticmethod
    def from_encoding_model(encoding_model, init_w=None, init_b=None, init_rules=None):
        logging.info('Creating LinearAndRuleModel from encoding model %s ', encoding_model.name)
        feat2id = dict()
        next_learned_id = 0
        next_rule_id = 0
        for feat in encoding_model.feature_list:
            if feat.name in encoding_model.must_link_rules or feat.name in encoding_model.must_not_link_rules:
                feat2id[feat.name] = (False, next_rule_id)
                next_rule_id += 1
            else:
                feat2id[feat.name] = (True, next_learned_id)
                next_learned_id += 1
        if init_w is None:
            init_w = np.ones(len([k for k in feat2id if feat2id[k][0]]), dtype=np.float32)
        if init_b is None:
            init_b = np.ones_like(init_w) * 1e-4
        if init_rules is None:
            init_rules = np.ones(len([k for k in feat2id if not feat2id[k][0]]), dtype=np.float32)
        for k, (is_learned, idx) in feat2id.items():
            if k in encoding_model.must_link_rules:
                assert not is_learned
                init_rules[idx] = must_link_rule
            elif k in encoding_model.must_not_link_rules:
                assert not is_learned
                init_rules[idx] = must_not_link_rule
            else:
                assert is_learned
        model = LinearAndRuleModel(init_w, init_b, init_rules, feat2id)
        return model
