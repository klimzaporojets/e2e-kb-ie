import torch
import torch.nn as nn

from models.misc.misc import batched_index_select


class SpanEndpoint(nn.Module):

    def __init__(self, dim_input, max_span_length, config):
        super(SpanEndpoint, self).__init__()
        self.span_embed = 'span_embed' in config
        self.dim_output = 2 * dim_input

        if self.span_embed:
            self.embed = nn.Embedding(max_span_length, config['span_embed'])
            self.dim_output += config['span_embed']

        if 'ff_dim' in config:
            self.ff = nn.Sequential(
                nn.Linear(self.dim_output, config['ff_dim']),
                nn.ReLU(),
                nn.Dropout(config['ff_dropout'])
            )
            self.dim_output = config['ff_dim']
        else:
            self.ff = nn.Sequential()

    def forward(self, inputs, b, e, max_width):
        b_vec = batched_index_select(inputs, b)
        e_vec = batched_index_select(inputs, torch.clamp(e, max=inputs.size(1) - 1))

        if self.span_embed:
            emb = self.embed(e - b)
            vec = torch.cat((b_vec, emb, e_vec), -1)
        else:
            vec = torch.cat((b_vec, e_vec), -1)

        return self.ff(vec)
