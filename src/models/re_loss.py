import torch
import torch.nn as nn

from metrics.misc import MetricObjective
from metrics.relations import MetricRelationF1x


def create_mapping(spans, clusters):
    num_batch = len(spans)
    max_spans = max([len(x) for x in spans])
    max_concepts = max([len(x) for x in clusters])

    mapping = torch.zeros(num_batch, max_concepts, max_spans)

    for batch, (myspans, myclusters) in enumerate(zip(spans, clusters)):
        span2index = {}
        for idx, span in enumerate(myspans):
            span2index[span] = idx

        for idx, cluster in enumerate(myclusters):
            for span in cluster:
                if span in span2index:  # in case relation pruner != coref pruner
                    mapping[batch, idx, span2index[span]] = 1.0

    return mapping


def sum_scores(scores, u):
    if scores.dim() != 4:
        raise BaseException("scores is not a 4-dimensional tensor")
    if u.dim() != 3:
        raise BaseException("mapping is not a 3-dimensional tensor")
    if scores.size(0) != u.size(0):
        raise BaseException("batch size doesn't match")
    num_batch, num_mentions, num_concepts = u.size()
    v = u.unsqueeze(1).expand(num_batch, num_concepts, num_mentions, num_concepts)
    o = torch.matmul(v, scores)
    p = o.view(o.size()[0:2] + (-1,))
    q = torch.matmul(u, p)
    q = q.view(q.size()[0:2] + o.size()[2:])
    return q


# TODO: move this to cpn utilities?
def decode_relations_new(targets, lengths, labels):
    relations = []
    for b, length in enumerate(lengths):
        rels = []
        for src, dst, rel in torch.nonzero(targets[b, 0:length, 0:length, :] > 0).tolist():
            rels.append((src, dst, labels[rel]))
        relations.append(rels)
    return relations


def create_relation_targets_2(pred_spans, relations, num_relations, span_lengths):
    gold_spans = relations['gold_spans']
    gold_m2i = relations['gold_m2i']
    gold_relations = relations['gold_relations']
    num_concepts = relations['num_concepts']

    num_batch = span_lengths.size(0)
    max_spans = span_lengths.max().item()

    targets = torch.zeros(num_batch, max_spans, max_spans, num_relations)

    for batch, (p_spans, g_spans, m2i, rels, max_clusters) in enumerate(
            zip(pred_spans, gold_spans, gold_m2i, gold_relations, num_concepts)):
        if len(rels) > 0:
            # max_clusters = max([max(src,dst) for src,dst,_ in rels])+1

            gold2index = {span: idx for span, idx in
                          zip(g_spans, m2i)}  # link gold span (mention) to gold index (entity)
            # max_cluster will be returned when span is not found
            pred2cluster = torch.LongTensor([gold2index.get(span, max_clusters) for span in p_spans])

            # cluster targets generates the ultime matrix -> cluster x cluster and a 1 for the correct relation
            rels = torch.LongTensor(rels)
            cluster_targets = torch.zeros(max_clusters + 1, max_clusters + 1, num_relations)
            cluster_targets[rels[:, 0], rels[:, 1], rels[:, 2]] = torch.ones(rels.size(0))

            # indices of the now found mentions which are mapped on entities -> if a wrong mention/ span -> max_cluster will result in 0 for sure
            dim = (pred2cluster.size(0), pred2cluster.size(0))
            r = pred2cluster.unsqueeze(-1).expand(dim).reshape(-1)
            c = pred2cluster.unsqueeze(-2).expand(dim).reshape(-1)

            indices = torch.arange(pred2cluster.size(0))
            rr = indices.unsqueeze(-1).expand(dim).reshape(-1)
            cc = indices.unsqueeze(-2).expand(dim).reshape(-1)
            # construct for the current spans: found concepts + relations
            targets[batch, rr, cc, :] = cluster_targets[r, c, :]

    return targets.to(span_lengths.device)


class LossRelationsX(nn.Module):

    def __init__(self, name, config, labels):
        super(LossRelationsX, self).__init__()
        self.name = name
        self.num_relations = len(labels)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.labels = labels
        self.enabled = config['enabled']
        self.weight = config['weight'] / len(self.labels) if config.get('normalize', True) else config['weight']
        self.debug = config['debug']
        print("LossRelationsX: weight={}".format(self.weight))

    def forward(self, relation_filtered, mention_scores, relations, coref, predict=False):
        output = {}

        span_lengths = relation_filtered['span_lengths']
        mention_mask = relation_filtered['square_mask']
        pred_spans = relation_filtered['spans']

        if self.enabled:
            mention_targets = create_relation_targets_2(pred_spans, relations, len(self.labels), span_lengths)
            obj = self.weight * (self.loss(mention_scores, mention_targets) * mention_mask.unsqueeze(
                -1)).sum() / self.num_relations
        else:
            obj = torch.tensor(0.0).cuda()

        output['loss'] = obj

        if self.enabled:
            mapping = create_mapping(pred_spans, coref['pred']).to(
                mention_scores.device)  # incase coref and relaion module use different pruner
            concept_targets = (sum_scores(mention_targets, mapping) > 0).float()  # to see if a relation is kept or not

            # only for debugging
            if mention_targets is not None:
                concept_lengths = [len(x) for x in coref['pred']]
                mytargets = decode_relations_new(concept_targets, concept_lengths, self.labels)
                output['target'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for
                                    clusters, triples in zip(coref['pred'], mytargets)]

            if predict:
                if mention_scores is None:
                    output['pred'] = [[] for x in coref['pred']]
                else:
                    # print('min:', mention_scores.min().item())
                    # print('max:', mention_scores.max().item())
                    pred_mentions = (mention_scores > 0).float()
                    pred_concepts = sum_scores(pred_mentions, mapping)
                    pred_concepts = (pred_concepts > 0).float()

                    concept_lengths = [len(x) for x in coref['pred']]
                    predictions = decode_relations_new(pred_concepts, concept_lengths, self.labels)
                    output['pred'] = [[(clusters[src], clusters[dst], rel) for src, dst, rel in triples] for
                                      clusters, triples in zip(coref['pred'], predictions)]

                output['gold'] = [[(clusters[src], clusters[dst], self.labels[rel]) for src, dst, rel in triples] for
                                  clusters, triples in zip(relations['gold_clusters2'], relations['gold_relations'])]
        else:
            output['pred'] = None
            output['gold'] = None

        return output['loss'], output

    def create_metrics(self):
        return [MetricRelationF1x(self.name, self.labels, verbose=self.debug),
                MetricObjective(self.name)] if self.enabled else []
