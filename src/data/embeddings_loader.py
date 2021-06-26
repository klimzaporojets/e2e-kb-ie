import gzip

import numpy as np
import torch

from training import settings


def load_wordembeddings(filename, accept={}, dim=300, out_of_voc_vector=None, init_random=False):
    # embedding_matrix = np.zeros((len(accept), dim))
    embedding_matrix = torch.zeros((len(accept), dim)).to(settings.device)
    if out_of_voc_vector is not None:
        print("WARNING: initialize word embeddings with ", out_of_voc_vector)
        embedding_matrix = embedding_matrix + out_of_voc_vector

    print("loading word vectors:", filename)
    found = 0

    file = gzip.open(filename, 'rt', encoding="utf8") if filename.endswith('.gz') else open(filename)
    lst_idxs_found = list()
    for line in file:
        # Note: use split(' ') instead of split() if you get an error.
        values = line.rstrip().split(' ')
        word = values[0]
        if word in accept:
            coefs = torch.tensor([float(v) for v in values[1:]], dtype=torch.float32)
            embedding_matrix[accept[word]] = coefs
            lst_idxs_found.append(accept[word])
            found += 1
    file.close()

    lst_idxs_not_found = list(sorted(set(range(embedding_matrix.shape[0])) - set(lst_idxs_found)))
    idxs_not_found = torch.tensor(lst_idxs_not_found, dtype=torch.int64).to(settings.device)
    if idxs_not_found.shape[0] > 0:
        if init_random:
            if len(lst_idxs_found) > 0:
                idxs_found = torch.tensor(lst_idxs_found, dtype=torch.int64).to(settings.device)
                embeddings_mean = torch.mean(embedding_matrix[idxs_found, :], dim=0)
                embeddings_std = torch.std(embedding_matrix[idxs_found, :], dim=0)
                embeddings_mean = embeddings_mean.unsqueeze(0).expand(idxs_not_found.shape[0], -1)
                embeddings_std = embeddings_std.unsqueeze(0).expand(idxs_not_found.shape[0], -1)
            else:
                embeddings_mean = torch.tensor(embedding_matrix.shape[1], dtype=torch.float32).to(settings.device)
                embeddings_std = torch.tensor(embedding_matrix.shape[1], dtype=torch.float32).to(settings.device)
                embeddings_mean[:] = 0.0
                embeddings_std[:] = 0.5
                embeddings_mean = embeddings_mean.unsqueeze(0).expand(idxs_not_found.shape[0], -1)
                embeddings_std = embeddings_std.unsqueeze(0).expand(idxs_not_found.shape[0], -1)
            nrml = torch.normal(embeddings_mean, embeddings_std)
            embedding_matrix[idxs_not_found, :] = nrml

    print("found: {} / {} = {}".format(found, len(accept), found / len(accept)))

    return embedding_matrix


def load_wordembeddings_with_random_unknowns(filename, accept={}, dim=300, debug=False, backoff_to_lowercase=False):
    print("loading word vectors:", filename)
    found = {}

    backoff = {}
    if backoff_to_lowercase:
        print("WARNING: backing off to lowercase")
        for x, idx in accept.items():
            backoff[x.lower()] = None
            backoff[x.casefold()] = None

    file = gzip.open(filename, 'rt', encoding="utf8") if filename.endswith('.gz') else open(filename, encoding="utf8")
    for line in file:
        values = line.rstrip().split(' ')
        word = values[0]
        if word in accept:
            found[accept[word]] = np.asarray(values[1:], dtype='float32')
        if word in backoff:
            backoff[word] = np.asarray(values[1:], dtype='float32')
    file.close()

    all_embeddings = np.asarray(list(found.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    embeddings = torch.FloatTensor(len(accept), dim).normal_(
        embeddings_mean, embeddings_std
    )
    for key, value in found.items():
        embeddings[key] = torch.FloatTensor(value)

    print("found: {} / {} = {}".format(len(found), len(accept), len(found) / len(accept)))
    print("words randomly initialized:", len(accept) - len(found))

    if debug:
        counter = 0
        for word in accept.keys():
            if accept[word] not in found:
                print("no such pretrained word: {} ({})".format(word, counter))
                counter += 1

    if backoff_to_lowercase:
        num_backoff = 0
        for word, idx in accept.items():
            if accept[word] not in found:
                if word.lower() in backoff and backoff[word.lower()] is not None:
                    print("backoff {} -> {}".format(word, word.lower()))
                    embeddings[idx, :] = torch.FloatTensor(backoff[word.lower()])
                    num_backoff += 1
                elif word.casefold() in backoff and backoff[word.casefold()] is not None:
                    print("casefold {} -> {}".format(word, word.lower()))
                    embeddings[idx, :] = torch.FloatTensor(backoff[word.casefold()])
                    num_backoff += 1
        print("num_backoff:", num_backoff)

    return embeddings


def load_wordembeddings_words(filename):
    words = []

    print("loading words:", filename)
    file = gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename)
    for line in file:
        # Note: use split(' ') instead of split() if you get an error.
        values = line.rstrip().split(' ')
        words.append(values[0])
    file.close()

    return words
