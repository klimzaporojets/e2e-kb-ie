import torch.nn as nn

from models.misc.misc import indices_to_spans, prune_spans, \
    create_masks, filter_spans


class MentionPruner(nn.Module):

    def __init__(self, dim_span, max_span_length, config):
        super(MentionPruner, self).__init__()
        self.config = config
        self.dim_span = dim_span
        self.hidden_dim = config['hidden_dim']  # 150
        self.hidden_dp = config['hidden_dropout']  # 0.4
        self.max_span_length = max_span_length
        self.sort_after_pruning = config['sort_after_pruning']
        self.prune_ratio = config['prune_ratio']

        print("MentionPruner:", self.max_span_length, self.prune_ratio, self.sort_after_pruning)

        self.scorer = nn.Sequential(
            nn.Linear(dim_span, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hidden_dp),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.hidden_dp),
            nn.Linear(self.hidden_dim, 1)
        )

    def create_new(self):
        return MentionPruner(self.dim_span, self.max_span_length, self.config)

    def forward(self, all_spans, sequence_lengths):
        span_vecs = all_spans['span_vecs']
        span_mask = all_spans['span_mask']
        span_begin = all_spans['span_begin']
        span_end = all_spans['span_end']

        prune_scores = self.scorer(span_vecs) - (1.0 - span_mask.unsqueeze(-1)) * 1e4
        span_pruned_indices, span_lengths = prune_spans(prune_scores, sequence_lengths, self.sort_after_pruning,
                                                        prune_ratio=self.prune_ratio)

        square_mask, triangular_mask = create_masks(span_lengths, span_pruned_indices.size(1))

        all_spans['span_scores'] = prune_scores

        return all_spans, {
            'prune_indices': span_pruned_indices,
            'span_vecs': filter_spans(span_vecs, span_pruned_indices),
            'span_scores': filter_spans(prune_scores, span_pruned_indices),
            'span_begin': filter_spans(span_begin.view(prune_scores.size()), span_pruned_indices),
            'span_end': filter_spans(span_end.view(prune_scores.size()), span_pruned_indices),
            'span_lengths': span_lengths,
            'square_mask': square_mask,
            'triangular_mask': triangular_mask,
            'spans': indices_to_spans(span_pruned_indices, span_lengths, self.max_span_length)
        }
