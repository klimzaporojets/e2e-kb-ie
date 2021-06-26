import torch
import torch.nn.utils.rnn as rnn_utils


def collate_scores(instances, max_candidates):
    max_spans = max([x.shape[0] for x in instances])

    output = torch.Tensor(len(instances), max_spans, max_candidates)
    output[:, :, :] = 0
    for b, instance in enumerate(instances):
        output[b, :instance.shape[0], :] = instance
    return output


def collate_candidates(instances, max_candidates):
    max_spans = max([x.shape[0] for x in instances])

    output = torch.LongTensor(len(instances), max_spans, max_candidates)
    output[:, :, :] = 0
    for b, instance in enumerate(instances):
        output[b, :instance.shape[0], :] = instance
    return output


def collate_spans(instances):
    max_spans = max([len(x) for x in instances])

    output = torch.LongTensor(len(instances), max_spans, 2)
    output[:, :, :] = 0
    for b, instance in enumerate(instances):
        for s, candidates in enumerate(instance):
            output[b, s, :] = torch.LongTensor(candidates)
    return output


def collate_character(batch, maxlen, padding, min_word_len=0):
    seqlens = [x.shape[0] for x in batch]
    max_word_len = max([curr_batch.shape[1] for curr_batch in batch])
    maxlen = min(maxlen, max_word_len)
    maxlen = max(maxlen, min_word_len)

    output = torch.LongTensor(len(batch), max(seqlens), maxlen)
    output[:, :, :] = padding
    for b, chars_tensor in enumerate(batch):
        output[b, :chars_tensor.shape[0], :chars_tensor.shape[1]] = chars_tensor
    return output


def main_collate(model, batch, device):
    # collate designed for batch size in 1; current architecture only supports batch size 1
    assert len(batch) == 1

    batch.sort(key=lambda x: x['tokens'].size()[0], reverse=True)

    sequence_lengths = torch.LongTensor([x['tokens'].size()[0] for x in batch])
    # TODO: move this to TextFieldEmbedderCharacters
    characters = collate_character([x['characters'] for x in batch], 50, model.embedder.char_embedder.padding,
                                   min_word_len=model.embedder.char_embedder.min_word_len)
    tokens = rnn_utils.pad_sequence([x['tokens'] for x in batch], batch_first=True)
    last_idx = max([len(x['tokens']) for x in batch]) - 1
    indices = rnn_utils.pad_sequence([x['tokens-indices'] for x in batch], batch_first=True, padding_value=last_idx)

    inputs = {
        'tokens': tokens.to(device),
        'characters': characters.to(device),
        'sequence_lengths': sequence_lengths.to(device),
        'token_indices': indices.to(device)
    }

    kb_entities = {}

    max_cands = model.kb_entities.max_nr_candidates + 1  # +1 because of NILL

    kb_entities['linker_candidates'] = \
        collate_candidates([x['candidates'] for x in batch], max_cands).to(torch.int64).to(device)

    kb_entities['len_candidates'] = batch[0]['len_candidates'].unsqueeze(0)

    kb_entities['linker_score_uniform'] = collate_scores([x['scores_uniform'] for x in batch], max_cands).to(device)

    kb_entities['linker_score_prior'] = collate_scores([x['scores_prior'] for x in batch], max_cands).to(device)

    kb_entities['gold_indices'] = rnn_utils.pad_sequence([x['gold_indices_words'] for x in batch],
                                                         batch_first=True).to(device)

    kb_entities['mask_candidates'] = collate_scores([x['mask_candidates'] for x in batch], max_cands).to(device)

    kb_entities['kb_empty'] = torch.zeros((1, 1, 1, 1)).to(device)

    inputs['kb_entities'] = kb_entities

    gold_spans = [[(m[0], m[1]) for m in x['spans']] for x in batch]

    idx_gold_spans = batch[0]['idx_gold_spans'].unsqueeze(0).to(device)

    gold_clusters = [x['gold_clusters'] for x in batch]

    metadata = {
        'identifiers': [x['id'] for x in batch],
        'tokens': [x['text'] for x in batch],
        'content': [x['content'] for x in batch],
        'begin': [x['begin'] for x in batch],
        'end': [x['end'] for x in batch]
    }
    inputs['gold_tags_indices'] = [x['gold_tags_indices'] for x in batch]
    inputs['gold_spans'] = gold_spans
    inputs['gold_m2i'] = [x['clusters'] for x in batch]

    inputs['spans_with_candidates_lst'] = [x['spans_with_candidates_lst'] for x in batch]

    inputs['gold_relations'] = [x['relations2'] for x in batch]
    inputs['num_concepts'] = [x['num_concepts'] for x in batch]
    inputs['gold_clusters2'] = gold_clusters
    inputs['idx_gold_spans'] = idx_gold_spans

    input_batch = {
        'input_batch': {
            'inputs': inputs,
            'metadata': metadata
        }
    }
    return input_batch
