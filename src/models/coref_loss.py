import torch
import torch.nn as nn

from metrics.coref import MetricCoref, MetricCorefAverage
from metrics.coref import decode_m2i, m2i_to_clusters
from metrics.misc import MetricObjective
from models.misc.misc import get_mask_from_sequence_lengths


def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def create_coref_target_forward(pred_spans, gold_spans, gold_clusters):
    num_batch = len(pred_spans)
    max_spans = max([len(x) for x in pred_spans])

    targets = torch.zeros(num_batch, max_spans, max_spans)

    for batch, (pred, gold, clusters) in enumerate(zip(pred_spans, gold_spans, gold_clusters)):
        gold2cluster = {}
        for idx, span in enumerate(gold):
            gold2cluster[span] = clusters[idx].item()

        for idx1, span1 in enumerate(pred):
            num_found = 0
            if span1 in gold2cluster:
                for idx2, span2 in enumerate(pred):
                    if idx2 < idx1 and span2 in gold2cluster and gold2cluster[span1] == gold2cluster[span2]:
                        targets[batch, idx1, idx2] = 1.0
                        num_found += 1

            if num_found == 0:
                targets[batch, idx1, idx1] = 1.0

    return targets


def convert(clusters, spans):
    out = [[spans[m] for m in cluster] for cluster in clusters]
    return out


# simplified
class LossCoref(nn.Module):

    def __init__(self, task, config):
        super(LossCoref, self).__init__()
        self.task = task
        self.weight = config.get('weight', 1.0)
        self.enabled = config['enabled']

    def forward(self, scores, gold_m2i, pred_spans, gold_spans, predict=False):
        output = {}

        if self.enabled:
            targets = create_coref_target_forward(pred_spans, gold_spans, gold_m2i).to(scores.device)

            if scores is not None:
                lengths = torch.LongTensor([len(x) for x in pred_spans]).to(scores.device)

                triangular_mask = torch.ones(scores.size()[1:]).tril(0).unsqueeze(0)
                constant = scores.max().item() + 100000
                additive_mask = (1 - triangular_mask) * -constant
                logits = torch.nn.functional.log_softmax(scores + additive_mask.to(scores.device), dim=-1)

            if scores is not None and targets is not None:
                loss = - logsumexp(logits + (1 - targets) * -100000)
                mask = get_mask_from_sequence_lengths(lengths, lengths.max().item()).float()
                output['loss'] = self.weight * (mask * loss).sum()
            else:
                raise BaseException("HUH")

            if predict:
                output['pred'] = [convert(m2i_to_clusters(x)[0], y) for x, y in
                                  zip(decode_m2i(logits, lengths), pred_spans)] if scores is not None else [[] for _ in
                                                                                                            pred_spans]
                output['gold'] = [convert(m2i_to_clusters(x.tolist())[0], y) for x, y in zip(gold_m2i, gold_spans)]
        else:
            output['loss'] = torch.tensor(0.0).cuda()
            output['pred'] = None
            output['gold'] = None

        return output['loss'], output

    def create_metrics(self):
        metrics = [
            MetricCoref(self.task, 'muc', MetricCoref.muc),
            MetricCoref(self.task, 'bcubed', MetricCoref.b_cubed, verbose=False),
            MetricCoref(self.task, 'ceafe', MetricCoref.ceafe, verbose=False),
        ] if self.enabled else []

        out = []
        out.extend(metrics)
        out.append(MetricCorefAverage(self.task, 'avg', metrics))
        out.append(MetricObjective(self.task))
        return out
