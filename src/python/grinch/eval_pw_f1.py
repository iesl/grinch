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


import itertools
import collections


def eval_macro_pw_f1(group2pred, group2gold):
    def clusters2dict(assgn):
        d = collections.defaultdict(list)
        for idx, c in enumerate(assgn):
            d[c].append(idx)
        return d

    scores = []
    assert len(group2pred) == len(group2gold)
    for pred, gold in zip(group2pred, group2gold):
        pred_pairs = set([x for c in clusters2dict(pred).values() for x in pairs_from_cluster(c)])
        gold_pairs = set([x for c in clusters2dict(gold).values() for x in pairs_from_cluster(c)])

        scores.append(pairwise_stats(pred_pairs, gold_pairs))

    prec = sum([x['micro_pw_prec'] for x in scores])
    rec = sum([x['micro_pw_rec'] for x in scores])
    f1 = 2.0 * prec * rec / (prec + rec)
    res = dict()
    res['macro_pw_prec'] = prec
    res['macro_pw_rec'] = rec
    res['macro_pw_f1'] = f1
    return res


def eval_micro_pw_f1(pred, gold, return_pairs=False):
    def clusters2dict(assgn):
        d = collections.defaultdict(list)
        for idx, c in enumerate(assgn):
            d[c].append(idx)
        return d

    pred_pairs = set([x for c in clusters2dict(pred).values() for x in pairs_from_cluster(c)])
    gold_pairs = set([x for c in clusters2dict(gold).values() for x in pairs_from_cluster(c)])
    return pairwise_stats(pred_pairs, gold_pairs, return_pairs)


def pairs_from_cluster(point_ids):
    return itertools.combinations(sorted(point_ids), 2)


def pairwise_stats(pred_pairs, gold_pairs, return_pairs=False):
    in_both = pred_pairs.intersection(gold_pairs)
    just_pred = pred_pairs.difference(gold_pairs)
    just_gold = gold_pairs.difference(pred_pairs)
    prec = len(in_both) / (len(in_both) + len(just_pred)) if (len(in_both) + len(just_pred)) > 0 else 0
    rec = len(in_both) / (len(in_both) + len(just_gold)) if (len(in_both) + len(just_gold)) > 0 else 0
    f1 = 2.0*prec*rec / (prec + rec) if (prec + rec) > 0 else 0
    res = dict()
    res['micro_pw_prec'] = prec
    res['micro_pw_rec'] = rec
    res['micro_pw_f1'] = f1

    res['micro_tp'] = len(in_both)
    res['micro_fp'] = len(just_pred)
    res['micro_fn'] = len(just_gold)
    if return_pairs:
        return res, in_both, just_pred, just_gold
    else:
        return res


def eval_ari(pred, gold):
    from sklearn.metrics import adjusted_rand_score
    res = dict()
    res['ari'] = adjusted_rand_score(pred, gold)
    return res


def eval_nmi(pred, gold):
    from sklearn.metrics import normalized_mutual_info_score
    res = dict()
    res['nmi'] = normalized_mutual_info_score(pred, gold)
    return res

if __name__ == "__main__":
    print(eval_micro_pw_f1([1, 1, 1, 1], [3, 2, 0, 0]))