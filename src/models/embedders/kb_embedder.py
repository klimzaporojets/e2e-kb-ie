import pandas as pd
import torch
import torch.nn as nn

from data.embeddings_loader import load_wordembeddings
from training import settings


class KBEmbedderAll(nn.Module):
    def __init__(self, dictionaries, config, span_extractor_dim):
        super(KBEmbedderAll, self).__init__()
        self.enabled = config['enabled']

        if self.enabled:
            self.dictionary = dictionaries['candidates']  # dict of KB entities
            self.dim = config['dim']  # size of embeddings
            self.h_dim = config['h_dim']  # size of hidden layer of nnet used for attention
            self.type = config['type']

            # self.enable_NER = config['NER'] if 'NER' in config else False
            self.max_nr_candidates = config['max_nr_candidates']

            self.init_random = config['init_random']
            self.spans_scope = config['spans_scope']

            self.filtered_dim = span_extractor_dim + self.dim
            if self.spans_scope == 'all':
                self.all_dim = span_extractor_dim + self.dim
            else:
                self.all_dim = span_extractor_dim

            self.embed = self.load_embeddings(config['embed_file'])
            if 'softmax' in config and config['softmax']:  # the softmax
                self.apply_softmax = True

            self.layers = None
            if self.type in {'attention'}:
                self.layers = nn.Sequential(  # The attention network
                    nn.Linear(span_extractor_dim + self.dim, self.h_dim),
                    nn.ReLU(),
                    nn.Linear(self.h_dim, 1)
                )
            elif self.type in {'attention_prior'}:
                self.layers = nn.Sequential(  # The attention network, the input also include + 1 for the prior
                    nn.Linear(span_extractor_dim + 1 + self.dim, self.h_dim),
                    nn.ReLU(),
                    nn.Linear(self.h_dim, 1)
                )
        else:
            self.filtered_dim = span_extractor_dim
            self.all_dim = span_extractor_dim

    def load_embeddings(self, filename):
        accept = {k: v for k, v in self.dictionary.word2idx.items()}
        words = load_wordembeddings(filename, accept=accept, dim=self.dim, init_random=self.init_random)

        # A simple lookup table that stores embeddings of a fixed dictionary and size.
        emb = nn.Embedding(self.dictionary.size, self.dim).to(settings.device)
        # to_copy = torch.Tensor(words).to(settings.device)
        emb.weight.data.copy_(words)  # put data in tensor
        emb.weight.requires_grad = False  # no change in embedding weights
        return emb

    def forward(self, filtered_spans, all_spans, inputs):
        candidates = inputs['kb_entities']['linker_candidates']
        if self.type == 'uniform':
            scores = inputs['kb_entities']['linker_score_uniform']
        else:
            scores = inputs['kb_entities']['linker_score_prior']

        len_candidates = inputs['kb_entities']['len_candidates'][0]
        if self.spans_scope == 'all':
            # if all possible spans, then selects all the spans with candidates
            idx_spans = (len_candidates > 0).nonzero().squeeze(-1)
        elif self.spans_scope == 'filtered':
            # if filtered, then selects the spans selected by the filter
            idx_spans = filtered_spans['prune_indices'][0]
        else:
            raise RuntimeError('no such spans_scope: ' + self.spans_scope)

        len_candidates = len_candidates[idx_spans]
        mask_spans_w_cands = len_candidates > 0
        idx_spans = idx_spans[mask_spans_w_cands]

        sel_candidates = candidates[0, idx_spans]
        if self.type == 'oracle_entities':
            ground_truth = inputs['kb_entities']['gold_indices'][0]
            idx_gold_spans = inputs['idx_gold_spans'][0]
            # the number of ground truth spans has to be equal to the number of candidates in ground_truth
            assert ground_truth.shape[-1] == idx_gold_spans.shape[-1]
            is_gold_candidate = torch.zeros(candidates.size(1), dtype=torch.bool).to(settings.device)
            is_gold_candidate[idx_gold_spans] = True
            gr_truth_cands_idx = torch.zeros(candidates.size(1), dtype=torch.int64).to(settings.device)
            gr_truth_cands_idx[is_gold_candidate] = ground_truth
            # if oracle then only the first candidate is chosen
            sel_candidates = sel_candidates[:, 1:2]  # not the first one because the first one is NIL
            # if oracle, then for ground truth spans, the ground truth candidate is assigned
            sel_candidates[is_gold_candidate[idx_spans]] = gr_truth_cands_idx[idx_spans][
                is_gold_candidate[idx_spans]].unsqueeze(-1)
            # assert to see that the changes actually happened
            assert torch.equal(sel_candidates[is_gold_candidate[idx_spans]],
                               gr_truth_cands_idx[idx_spans][is_gold_candidate[idx_spans]].unsqueeze(-1))

        sel_candidates = self.embed(sel_candidates)

        # if there are candidates
        if mask_spans_w_cands.sum() > 0:
            sel_mask_candidates = inputs['kb_entities']['mask_candidates'][0, idx_spans]
            if self.type in {'attention_prior', 'attention'}:
                if self.spans_scope == 'filtered':
                    sel_spans_w_cands = filtered_spans['span_vecs'][0][mask_spans_w_cands]
                else:
                    # sel_spans_w_cands = all_spans['span_vecs'][0][mask_spans_w_cands]
                    sel_spans_w_cands = all_spans['span_vecs'][0].view(-1, all_spans['span_vecs'].shape[-1])[idx_spans]

                sel_spans_w_cands = sel_spans_w_cands.unsqueeze(-2).expand(-1, sel_candidates.shape[1], -1)
                input_to_att = [sel_spans_w_cands, sel_candidates]
                if self.type == 'attention_prior':
                    input_to_att.append(scores[0, idx_spans].unsqueeze(-1))
                ment_cand = torch.cat(input_to_att, dim=-1)
                scores = self.layers(ment_cand).squeeze(-1)
                if self.apply_softmax:
                    masked_scores = scores + torch.log(sel_mask_candidates)
                    scores = masked_scores.softmax(dim=-1)
                sel_cands_w_spans = (sel_candidates * scores.unsqueeze(-1)).sum(dim=1)
                # sel_cand_w_spans.shape --> [11,400]
            elif self.type in {'prior', 'uniform'}:
                sel_scores = scores[0, idx_spans]
                sel_cands_w_spans = (sel_candidates * sel_scores.unsqueeze(-1)).sum(dim=1)
            elif self.type == 'oracle_entities':
                sel_cands_w_spans = sel_candidates.squeeze(1)
            else:
                raise RuntimeError('Not recognized self.type in KBEmbedderAll ' + self.type)

            all_spans['span_vecs'] = torch.nn.functional.pad(input=all_spans['span_vecs'], pad=(0, self.dim),
                                                             mode="constant", value=0.0)
            if self.spans_scope == 'filtered':
                filtered_spans['span_vecs'] = torch.nn.functional.pad(input=filtered_spans['span_vecs'],
                                                                      pad=(0, self.dim),
                                                                      mode='constant', value=0.0)
                filtered_spans['span_vecs'][0][mask_spans_w_cands, -self.dim:] = sel_cands_w_spans
                # controls that the transfer has been actually done
                assert torch.equal(filtered_spans['span_vecs'][0][mask_spans_w_cands, -self.dim:], sel_cands_w_spans)

            all_spans_plain = all_spans['span_vecs'].view(-1, all_spans['span_vecs'].shape[-1])
            all_spans_plain[idx_spans, -self.dim:] = sel_cands_w_spans

            # controls that the transfer has been actually done
            to_check_assert = all_spans['span_vecs'].view(-1, all_spans['span_vecs'].shape[-1])[idx_spans, -self.dim:]
            assert torch.equal(to_check_assert, sel_cands_w_spans)
        else:
            if self.spans_scope == 'filtered':
                filtered_spans['span_vecs'] = torch.nn.functional.pad(input=filtered_spans['span_vecs'],
                                                                      pad=(0, self.dim),
                                                                      mode="constant", value=0.0)
            all_spans['span_vecs'] = torch.nn.functional.pad(input=all_spans['span_vecs'], pad=(0, self.dim),
                                                             mode="constant", value=0.0)
        if self.spans_scope == 'filtered':
            return filtered_spans['span_vecs'], all_spans['span_vecs']
        else:
            return all_spans['span_vecs']


def read_json_file(name):
    """ A function that reads the look-up table from the json file """
    df = pd.read_json(name, lines=True)
    df['candidates'] = df['candidates'].apply(lambda x: x[:17])
    df['scores'] = df['scores'].apply(lambda x: x[:17])
    df.index = df['text']
    dictionary = df.to_dict('index')
    # dictionary={}
    # with open(name, 'r', encoding='utf8') as fp:
    #     for line in fp:
    #         data = json.loads(line)
    #         dictionary[data['text']] = data
    return dictionary
