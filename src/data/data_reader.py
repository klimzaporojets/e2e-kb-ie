import json
import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data.tokenizer import TokenizerCPN
from models.embedders.kb_embedder import read_json_file


def collate_character(chars_in_document, maxlen, padding, min_word_len=0):
    # seqlens = [len(x) for x in batch]
    seqlens = len(chars_in_document)
    max_word_len = max([len(w) for w in chars_in_document])
    maxlen = min(maxlen, max_word_len)
    maxlen = max(maxlen, min_word_len)

    output = torch.LongTensor(seqlens, maxlen)
    output[:, :] = padding
    # for i, sentence in enumerate(batch):
    for pos, token in enumerate(chars_in_document):
        token_len = len(token)
        if token_len < maxlen:
            output[pos, :len(token)] = torch.from_numpy(np.array(token, dtype=np.long))
        else:
            output[pos, :] = torch.from_numpy(np.array(token[0:maxlen], dtype=np.long))
    return output


def collate_scores(spans_w_scores, max_candidates):
    nr_spans = len(spans_w_scores)

    output = torch.Tensor(nr_spans, max_candidates)
    output[:, :] = 0
    # for b, instance in enumerate(spans_w_candidates):
    # num_spans = len(instance)
    for s, candidate_scores in enumerate(spans_w_scores):
        nr_candidates = len(candidate_scores)
        output[s, :nr_candidates] = torch.Tensor(candidate_scores)
    return output


def collate_candidates(spans_w_candidates, max_candidates):
    nr_spans = len(spans_w_candidates)

    output = torch.LongTensor(nr_spans, max_candidates)
    lengths = torch.LongTensor(nr_spans)
    output[:, :] = 0
    lengths[:] = 0
    # for b, instance in enumerate(spans_w_candidates):
    # num_spans = len(instance)
    for s, candidates in enumerate(spans_w_candidates):
        nr_candidates = len(candidates)
        output[s, :nr_candidates] = torch.LongTensor(candidates)
        lengths[s] = nr_candidates
    return output, lengths


def create_datasets(config, dictionaries):
    if config['dataloader']['type'] == 'cpn':
        datasets = {name: DatasetCPN(name, value, config, dictionaries) for name, value in config['datasets'].items()}
    else:
        raise BaseException("no such data loader:", config['dataloader'])

    train = datasets[config['trainer']['train']]  # train dataset
    train.train = True
    evaluate = config['trainer']['evaluate']  # test dataset

    return datasets, train, evaluate


def get_token_buckets(tokens):
    token2idx = {}
    for token in tokens:
        token = token.lower()
        if token not in token2idx:
            token2idx[token] = len(token2idx)
    return [token2idx[token.lower()] for token in tokens]


class DatasetCPN(Dataset):

    def __init__(self, name, dataset_config, config, dictionaries):
        self.name = name
        self.tokenize = dataset_config['tokenize']
        self.tag = dataset_config['tag']
        self.dict_words = dictionaries['words']
        self.dict_characters = dictionaries['characters']
        self.dict_tags = dictionaries['tags-y']
        self.dict_relations = dictionaries['relations']
        self.dict_entities = dictionaries.get('entities', None)

        # added:  dictionaries for the KB relations and the candidates from the dictionary
        self.dict_candidates = dictionaries.get('candidates', None)
        self.dict_kb_relations = dictionaries.get('kb_relations', None)

        self.max_span_length = config['model']['max_span_length']
        if 'kb_embedder' in config['model']:
            self.max_nr_candidates = config['model']['kb_embedder']['max_nr_candidates']

        self.char_padding = dictionaries['characters'].lookup('PADDING')

        self.min_word_len = config['model']['text_embedder']['char_embedder']['min_word_len']

        if self.tokenize:
            self.tokenizer = TokenizerCPN()
        path = dataset_config['filename']

        self.instances = []
        self.number_of_lost_mentions = 0

        # added: read the dictionary file into memory
        if 'json' in dataset_config:
            kb_entities = read_json_file(dataset_config['json'])
        else:
            kb_entities = None
        self.kb_entities = kb_entities

        print("Loading {} ({}) tokenize={} tag={}".format(path, name, self.tokenize, self.tag))
        for filename in tqdm(os.listdir(path)):
            self.load(os.path.join(path, filename))
        print("done.")

        print("Number of instances in {}: {}.".format(self.name, len(self)))
        # print("Number of mentions lost due to tokenization: {}".format(self.number_of_lost_mentions))
        self.print_histogram_of_span_length()

    def print_histogram_of_span_length(self):
        counter = Counter()
        total = 0
        fail = 0
        for instance in self.instances:
            for begin, end in instance['spans']:
                if begin is None or end is None:
                    fail += 1
                else:
                    counter[end - begin] += 1
                    total += 1

        print("span\tcount\trecall")
        cum = 0
        for span_length in sorted(counter.keys()):
            count = counter[span_length]
            cum += count
            print("{}\t{}\t{}".format(span_length, count, cum / total))
        print()
        print("failed spans:", fail)

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            mentions = data['mentions']
            concepts = data['concepts']

            if self.tag not in data['tags']:
                return

            if self.tokenize:
                tokens = self.tokenizer.tokenize(data['content'])
                begin = [token['offset'] for token in tokens]
                end = [token['offset'] + token['length'] for token in tokens]
                tokens = [token['token'] for token in tokens]
            else:
                tokens = data['tokenization']['tokens']
                begin = data['tokenization']['begin']
                end = data['tokenization']['end']

            if len(tokens) == 0:
                print("WARNING: dropping empty document")
                return

            begin_to_index = {pos: idx for idx, pos in enumerate(begin)}
            end_to_index = {pos: idx for idx, pos in enumerate(end)}

            # this makes life easier
            for concept in concepts:
                concept['mentions'] = []
            for mention in mentions:
                concept = concepts[mention['concept']]
                mention['concept'] = concept
                mention['token_begin'] = begin_to_index.get(mention['begin'], None)
                mention['token_end'] = end_to_index.get(mention['end'], None)
                if mention['token_begin'] is None or mention['token_end'] is None:
                    self.number_of_lost_mentions += 1
                concept['mentions'].append(mention)

            identifier = data['id']
            token_indices = self.get_token_indices(tokens)
            character_indices = self.get_character_indices(tokens)

            # passes the indices to torch tensor
            character_indices = collate_character(character_indices, 50, self.char_padding,
                                                  min_word_len=self.min_word_len)

            spans = [(mention['token_begin'], mention['token_end']) for mention in
                     data['mentions']]  # token begin, is the token with which the mention starts
            gold_clusters = [[(mention['token_begin'], mention['token_end']) for mention in concept['mentions']] for
                             concept in concepts]
            spans_to_mentions = \
                {(mention['token_begin'], mention['token_end']): mention for mention in data['mentions']}

            # Added: reading candidates from the dictionary file
            spans_with_candidates, len_candidates, candidates, scores, scores_uniform, mask_soft, idx_gold_spans, \
            gold_indices_words = self.read_mention_dict(data['content'], tokens, begin, end, spans_to_mentions)

        # +1 because of NILL
        candidates, _ = collate_candidates(candidates, self.max_nr_candidates + 1)
        # +1 because of NILL
        scores = collate_scores(scores, self.max_nr_candidates + 1)
        # +1 because of NILL
        scores_uniform = collate_scores(scores_uniform, self.max_nr_candidates + 1)
        # +1 because of NILL
        mask_soft = collate_scores(mask_soft, self.max_nr_candidates + 1)

        len_candidates = torch.LongTensor(len_candidates)

        gold_indices_words = torch.LongTensor(gold_indices_words)

        self.instances.append({
            'id': identifier,  # ID of the article in the dataset
            'content': data['content'],  # The article in text format
            'begin': torch.IntTensor(begin),  # tensor with all begin positions of the words
            'end': torch.IntTensor(end),  # tensor with al the end positions
            'tokens': torch.LongTensor(token_indices),
            'characters': character_indices,  # id's for all the CHARACTERS, in sub array grouped by word
            'tokens-indices': torch.LongTensor(get_token_buckets(tokens)),
            'spans': spans,
            'idx_gold_spans': torch.LongTensor(idx_gold_spans),
            'gold_clusters': gold_clusters,
            'gold_tags_indices': self.get_span_tags(mentions),
            'text': tokens,
            'clusters': torch.IntTensor([mention['concept']['concept'] for mention in mentions]),
            'relations2': self.get_relations(data),
            'num_concepts': len(concepts),
            'gold_indices_words': gold_indices_words,
            'scores_uniform': scores_uniform,
            'mask_candidates': mask_soft,
            'scores_prior': scores,
            'candidates': candidates,
            'len_candidates': len_candidates,
            'spans_with_candidates_lst': spans_with_candidates
        })

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)

    def get_token_indices(self, tokens):
        return [self.dict_words.lookup(token) for token in tokens]

    def get_character_indices(self, tokens):
        output = []
        for token in tokens:
            token = '<' + token + '>'
            output.append([self.dict_characters.lookup(c) for c in token])
        return output

    def get_span_tags(self, mentions):
        spans = []
        for mention in mentions:
            if mention['token_begin'] is not None and mention['token_end'] is not None:
                spans.extend([(mention['token_begin'], mention['token_end'], self.dict_tags.lookup(tag)) for tag in
                              mention['concept']['tags']])
        return spans

    def get_relations(self, data):
        return [(relation['s'], relation['o'], self.dict_relations.lookup(relation['p'])) for relation in
                data['relations']]

    def get_linker_candidates(self, data):
        if 'wiki::improve_algo' in data['tags']:
            # if 'annotation::links' in data['tags']:
            candidates = []

            for mention in data['mentions']:
                if mention['token_begin'] is not None and mention['token_end'] is not None:
                    cs = ['NILL'] + (mention['candidates'] if 'candidates' in mention else [])
                    candidates.append([self.dict_entities.add(c) for c in cs])
                else:
                    candidates.append([])
            if candidates == [] or candidates == [[]]:
                candidates = [[0]]

        else:
            candidates = [[] for mention in data['mentions']]

        return candidates

    def get_linker_gold_indices_endtoend(self, data):
        """Collect the gold link candidates for the gold mentions for the DICTIONARY.
            # Arguments
                data: a dictionary containing an article from the dataset.
            # Returns
                gold: a list of gold link candidates for the gold mentions.
            """
        if self.kb_entities is not None:
            gold = []
            for mention in data['mentions']:
                concept = mention['concept']
                if 'link' in concept:
                    if concept['link'] is not None and concept['link'] != 'NILL':
                        gold.append((mention['token_begin'], mention['token_end'],
                                     self.dict_candidates.lookup(concept['link'])))
                    else:
                        gold.append((mention['token_begin'], mention['token_end'], 0))
                else:
                    gold.append((mention['token_begin'], mention['token_end'], 0))
            return gold
        return None

    def read_mention_dict(self, content, tokens, begin, end, spans_to_mentions):

        self.dict_candidates.lookup('NILL')
        spans_with_candidates = []
        candidates = []
        scores = []
        scores_uniform = []
        mask_soft = []
        dictionary = self.kb_entities
        len_candidates = []
        span_begin = [[i for _ in range(self.max_span_length)] for i in range(len(tokens))]
        span_end = [[i + j for j in range(self.max_span_length)] for i in range(len(tokens))]
        span_mask = [[1 if k < len(tokens) else 0 for k in j] for j in span_end]
        idx_gold_spans = list()
        gold_indices_words = list()
        for i in range(len(tokens)):
            for j in range(self.max_span_length):
                if span_mask[i][j] == 1:
                    span_begin_idx = begin[span_begin[i][j]]
                    span_end_idx = end[span_end[i][j]]

                    span_text = content[span_begin_idx:span_end_idx]
                    if span_text in dictionary:
                        span_candidates = dictionary[span_text]['candidates'][0:self.max_nr_candidates]

                        if (span_begin[i][j], span_end[i][j]) in spans_to_mentions:
                            idx_gold_spans.append(i * self.max_span_length + j)
                            concept = spans_to_mentions[(span_begin[i][j], span_end[i][j])]['concept']
                            # concept = concepts[concept_id]
                            if 'link' in concept:
                                if concept['link'] is not None and concept['link'] != 'NILL' \
                                        and concept['link'] in set(span_candidates):
                                    gold_indices_words.append(self.dict_candidates.lookup(concept['link']))
                                else:
                                    gold_indices_words.append(0)
                            else:
                                gold_indices_words.append(0)

                        spans_with_candidates.append((span_begin[i][j], span_end[i][j]))
                        nr_span_candidates = len(span_candidates)
                        nr_masked_candidates = self.max_nr_candidates - nr_span_candidates
                        span_cand_scores = dictionary[span_text]['scores'][0:self.max_nr_candidates]

                        nill_id = self.dict_candidates.lookup('NILL')
                        curr_cand_append = [nill_id] + [self.dict_candidates.lookup(c) for c in span_candidates]
                        candidates.append(curr_cand_append)
                        len_candidates.append(len(curr_cand_append))
                        # + 1 because of NILL
                        curr_mask_soft = [1.0] * (nr_span_candidates + 1) + [0.0] * nr_masked_candidates
                        assert len(curr_mask_soft) == self.max_nr_candidates + 1
                        mask_soft.append(curr_mask_soft)
                        uniform_weight = 1.0 / nr_span_candidates
                        curr_scores_uniform = [0.0] + [uniform_weight] * nr_span_candidates
                        scores_uniform.append(curr_scores_uniform)
                        scores.append([0] + span_cand_scores)
                    else:
                        # (kzaporoj) - added here to make sizes compatible
                        scores.append([0.0])
                        scores_uniform.append([0.0])
                        mask_soft.append([0.0])
                        len_candidates.append(0)
                        candidates.append([0])
                        # if not in the dictionary, but still a valid gold span, consider the gold link to be NIL
                        # (no candidates)
                        if (span_begin[i][j], span_end[i][j]) in spans_to_mentions:
                            gold_indices_words.append(0)
                            idx_gold_spans.append(i * self.max_span_length + j)
                else:
                    scores.append([0.0])
                    scores_uniform.append([0.0])
                    mask_soft.append([0.0])
                    len_candidates.append(0)
                    candidates.append([0])
        if scores == [] or scores == [[]]:
            scores = [[0]]
        if candidates == [] or candidates == [[]]:
            candidates = [[0]]
        return spans_with_candidates, len_candidates, candidates, scores, scores_uniform, mask_soft, idx_gold_spans, \
               gold_indices_words
