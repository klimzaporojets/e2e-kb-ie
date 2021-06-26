import torch.nn as nn

# RelProp without a separate pruner
from models.misc.misc import OptFFpairs


class ModuleRelScorer(nn.Module):

    def __init__(self, dim_span, span_pair_generator, labels, config):
        super(ModuleRelScorer, self).__init__()
        self.rel_prop = config['rel_prop']

        print("ModuleRelProp(rp={})".format(self.rel_prop))

        self.scorer = OptFFpairs(dim_span, len(labels), config, span_pair_generator)
        self.A = nn.Linear(len(labels), dim_span, bias=False)

    def forward(self, all_spans, filtered_spans):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        relation_scores = self.scorer(update, filtered_span_begin, filtered_span_end)

        return update_all, update_filtered, relation_scores
