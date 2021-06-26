import torch.nn as nn

from models.misc.misc import coref_add_scores, OptFFpairs


class ModuleCorefScorer(nn.Module):

    def __init__(self, dim_span, coref_pruner, span_pair_generator, config):
        super(ModuleCorefScorer, self).__init__()
        self.coref_prop = config['coref_prop']
        self.update_coref_scores = config['update_coref_scores']

        print("ModuleCorefProp(cp={})".format(self.coref_prop))

        self.coref_pruner = coref_pruner
        self.coref = OptFFpairs(dim_span, 1, config, span_pair_generator)

    def forward(self, all_spans, filtered_spans, sequence_lengths):
        update = filtered_spans['span_vecs']
        filtered_span_begin = filtered_spans['span_begin']
        filtered_span_end = filtered_spans['span_end']

        coref_scores = self.coref(update, filtered_span_begin, filtered_span_end).squeeze(-1)
        coref_scores = coref_add_scores(coref_scores, filtered_spans['span_scores'])

        update_all = all_spans.copy()
        update_filtered = filtered_spans.copy()

        return update_all, update_filtered, coref_scores
