from collections import Counter

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


## MOSTLY COPIED FROM ALLENNLP

def decode_m2i(scores, lengths):
    output = []
    for b, length in enumerate(lengths.tolist()):
        m2i = list(range(length))
        if length > 0:
            # print('scores:', length, scores[b, 0:length, :])
            # _, indices = torch.max(scores[b, 0:length, 0:length], -1)
            _, indices = torch.max(scores[b, 0:length, :], -1)
            for src, dst in enumerate(indices.tolist()):
                if src < len(m2i) and dst < len(m2i):
                    m2i[src] = m2i[dst]
                else:
                    # sanity check: this should never ever happen !!!
                    print("ERROR: invalid index")
                    print("length:", length)
                    print("scores:", scores[b, 0:length, :])
                    print("scores:", scores.min().item(), scores.max().item())
                    print("indices:", indices)
                    print("LENGTHS:", lengths)
                    exit(1)
        output.append(m2i)
    return output


def m2i_to_clusters(m2i):
    clusters = {}
    m2c = {}
    for m, c in enumerate(m2i):
        if c not in clusters:
            clusters[c] = []
        clusters[c].append(m)
        m2c[m] = clusters[c]
    return list(clusters.values()), m2c


def mention2cluster(clusters):
    clusters = [tuple(tuple(m) for m in gc) for gc in clusters]
    mention_to_cluster = {}
    for cluster in clusters:
        for mention in cluster:
            mention_to_cluster[mention] = cluster
    return mention_to_cluster


class MetricCoref:

    def __init__(self, task, name, m, verbose=False):
        self.task = task
        self.name = name
        self.m = m
        self.iter = 0
        self.clear()
        self.max_iter = 0
        self.max_f1 = 0
        self.verbose = verbose

    def clear(self):
        self.precision_numerator = 0
        self.precision_denominator = 0
        self.recall_numerator = 0
        self.recall_denominator = 0

    def step(self):
        self.clear()
        self.iter += 1

    def update(self, scores, targets, args, metadata):
        gold_m2is, lengths, mentions = args['gold_m2is'], args['lengths'], args['mentions']
        pred_m2is = decode_m2i(scores, lengths)
        gold_m2is = [x.tolist() for x in gold_m2is]
        for pred_m2i, gold_m2i, identifier, ms in zip(pred_m2is, gold_m2is, metadata['identifiers'], mentions):
            pred_clusters, _ = m2i_to_clusters(pred_m2i)
            gold_clusters, _ = m2i_to_clusters(gold_m2i)

            p_num, p_den = self.m(pred_clusters, {k: (v, v) for k, v in enumerate(gold_m2i)})
            r_num, r_den = self.m(gold_clusters, {k: (v, v) for k, v in enumerate(pred_m2i)})

            if self.verbose:
                print("ID", identifier)
                print('pred_clusters:', pred_clusters)
                print('gold_clusters:', gold_clusters)
                print('precision: {} / {}'.format(p_num, p_den))
                print('recall:    {} / {}'.format(r_num, r_den))
                print()

            self.precision_numerator += p_num
            self.precision_denominator += p_den
            self.recall_numerator += r_num
            self.recall_denominator += r_den

    def add(self, pred, gold):
        if self.m == self.ceafe:
            p_num, p_den, r_num, r_den = self.m(pred, gold)
        else:
            p_num, p_den = self.m(pred, mention2cluster(gold))
            r_num, r_den = self.m(gold, mention2cluster(pred))

        self.precision_numerator += p_num
        self.precision_denominator += p_den
        self.recall_numerator += r_num
        self.recall_denominator += r_den

    def update2(self, output_dict, metadata):
        # print("UPDATE2")
        # print("pred:", output_dict["pred"])
        # print("gold:", output_dict["gold"])
        # print()
        # print("clusters:", output_dict["clusters"])
        # print("coref_gold:", output_dict["coref_gold"])

        for pred, gold, identifier, tokens in zip(output_dict["pred"], output_dict["gold"], metadata['identifiers'],
                                                  metadata['tokens']):
            if self.m == self.ceafe:
                p_num, p_den, r_num, r_den = self.m(pred, gold)
            else:
                p_num, p_den = self.m(pred, mention2cluster(gold))
                r_num, r_den = self.m(gold, mention2cluster(pred))

            if self.verbose:
                print("ID", identifier)
                print("pred:", pred)
                print("gold:", gold)
                print("pred:", [[' '.join(tokens[begin:(end + 1)]) for begin, end in cluster] for cluster in pred])
                print("gold:", [[' '.join(tokens[begin:(end + 1)]) for begin, end in cluster] for cluster in gold])
                print('precision: {} / {}'.format(p_num, p_den))
                print('recall:    {} / {}'.format(r_num, r_den))
                print()

            self.precision_numerator += p_num
            self.precision_denominator += p_den
            self.recall_numerator += r_num
            self.recall_denominator += r_den

    def get_f1(self):
        precision = 0 if self.precision_denominator == 0 else \
            self.precision_numerator / float(self.precision_denominator)
        recall = 0 if self.recall_denominator == 0 else \
            self.recall_numerator / float(self.recall_denominator)
        return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def print(self, dataset_name, details=False):
        f1 = self.get_f1()
        if f1 > self.max_f1:
            self.max_f1 = f1
            self.max_iter = self.iter

        print('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}\tmax-iter: {}\tmax-{}-f1: {}'
              '\tstall: {}'.format(dataset_name,
                                   self.task,
                                   self.iter,
                                   self.name,
                                   f1,
                                   self.max_iter,
                                   self.name,
                                   self.max_f1,
                                   self.iter - self.max_iter))

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('{}/{}-f1'.format(dataset_name, self.name), self.get_f1(), self.iter)

    @staticmethod
    def b_cubed(clusters, mention_to_gold):
        numerator, denominator = 0, 0
        for cluster in clusters:
            if len(cluster) == 1:
                continue
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                if len(cluster2) != 1:
                    correct += count * count
            numerator += correct / float(len(cluster))
            denominator += len(cluster)
        return numerator, denominator

    @staticmethod
    def muc(clusters, mention_to_gold):
        true_p, all_p = 0, 0
        for cluster in clusters:
            all_p += len(cluster) - 1
            true_p += len(cluster)
            linked = set()
            for mention in cluster:
                if mention in mention_to_gold:
                    linked.add(mention_to_gold[mention])
                else:
                    true_p -= 1
            true_p -= len(linked)
        return true_p, all_p

    @staticmethod
    def phi4(gold_clustering, predicted_clustering):
        """
        Subroutine for ceafe. Computes the mention F measure between gold and
        predicted mentions in a cluster.
        """
        return (
                2
                * len([mention for mention in gold_clustering if mention in predicted_clustering])
                / float(len(gold_clustering) + len(predicted_clustering))
        )

    @staticmethod
    def ceafe(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.
        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        """
        clusters = [cluster for cluster in clusters if len(cluster) != 1]
        gold_clusters = [cluster for cluster in gold_clusters if len(cluster) != 1]  # is this really correct?
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = MetricCoref.phi4(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        return similarity, len(clusters), similarity, len(gold_clusters)


class MetricCorefAverage:

    def __init__(self, task, name, metrics):
        self.task = task
        self.name = name
        self.metrics = metrics
        self.iter = 0
        self.max_f1 = 0
        self.max_iter = 0

    def step(self):
        self.iter += 1

    def update2(self, output_dict, metadata):
        return

    def get_f1(self):
        scores = [x.get_f1() for x in self.metrics]
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0

    def print(self, dataset_name, details=False):
        f1 = self.get_f1()

        if f1 > self.max_f1:
            self.max_f1 = f1
            self.max_iter = self.iter

        print('EVAL-COREF\t{}-{}\tcurr-iter: {}\t{}-f1: {}\tmax-iter: {}\tmax-{}-f1: {}'
              '\tstall: {}'.format(dataset_name,
                                   self.task,
                                   self.iter,
                                   self.name,
                                   f1,
                                   self.max_iter,
                                   self.name,
                                   self.max_f1,
                                   self.iter - self.max_iter))

    def log(self, tb_logger, dataset_name):
        tb_logger.log_value('{}/{}-f1'.format(dataset_name, self.name), self.get_f1(), self.iter)
