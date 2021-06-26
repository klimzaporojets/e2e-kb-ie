import numpy as np
import torch
import torch.nn as nn

from data.embeddings_loader import load_wordembeddings, load_wordembeddings_words, \
    load_wordembeddings_with_random_unknowns
from models.misc.misc import CNNMaxpool


class TextEmbedder(nn.Module):
    def __init__(self, dictionaries, config):
        super(TextEmbedder, self).__init__()
        self.config = config
        self.dim_output = 0
        if 'char_embedder' in config:
            self.char_embedder = TextFieldEmbedderCharacters(dictionaries, config['char_embedder'])
            self.dim_output += self.char_embedder.dim_output
        if 'text_field_embedder' in config:
            self.word_embedder = TextFieldEmbedderTokens(dictionaries, config['text_field_embedder'])
            self.dim_output += self.word_embedder.dim

    def forward(self, characters, tokens):
        outputs = []
        if 'char_embedder' in self.config:
            outputs.append(self.char_embedder(characters))
        if 'text_field_embedder' in self.config:
            outputs.append(self.word_embedder(tokens))

        return torch.cat(outputs, -1)


class TextFieldEmbedderList(nn.Module):
    def __init__(self, list):
        super(TextFieldEmbedderList, self).__init__()
        self.embedders = nn.ModuleList(list)
        self.dim = sum([embedder.dim for embedder in self.embedders])

    def forward(self, inputs):
        outputs = [embedder(inputs) for embedder in self.embedders]
        return torch.cat(outputs, 1)


class TextFieldEmbedderTokens(nn.Module):

    def __init__(self, dictionaries, config):
        super(TextFieldEmbedderTokens, self).__init__()
        self.dictionary = dictionaries[config['dict']]
        self.dim = config['dim']
        self.embed = nn.Embedding(self.dictionary.size, self.dim)
        self.dropout = nn.Dropout(config['dropout'], inplace=True)
        self.normalize = 'norm' in config
        self.freeze = config.get('freeze', False)

        if 'embed_file' in config:
            self.init_unknown = config['init_unknown']
            self.init_random = config['init_random']
            self.backoff_to_lowercase = config['backoff_to_lowercase']

            self.load_embeddings(config['embed_file'])
        else:
            print("WARNING: training word vectors from scratch")

        nrms = self.embed.weight.norm(p=2, dim=1, keepdim=True)
        print("norms: min={} max={} avg={}".format(nrms.min().item(), nrms.max().item(), nrms.mean().item()))

    def load_all_wordvecs(self, filename):
        print("LOADING ALL WORDVECS")
        words = load_wordembeddings_words(filename)
        for word in words:
            self.dictionary.add(word)
        self.load_embeddings(filename)
        print("DONE")

    def load_embeddings(self, filename):
        if self.init_random:
            embeddings = load_wordembeddings_with_random_unknowns(filename, accept=self.dictionary.word2idx,
                                                                  dim=self.dim,
                                                                  backoff_to_lowercase=self.backoff_to_lowercase)
        else:
            unknown_vec = np.ones((self.dim)) / np.sqrt(self.dim) if self.init_unknown else None

            word_vectors = load_wordembeddings(filename, accept=self.dictionary.word2idx, dim=self.dim,
                                               out_of_voc_vector=unknown_vec)
            if self.normalize:
                norms = np.einsum('ij,ij->i', word_vectors, word_vectors)
                np.sqrt(norms, norms)
                norms += 1e-8
                word_vectors /= norms[:, np.newaxis]

            embeddings = torch.from_numpy(word_vectors)

        device = next(self.embed.parameters()).device
        self.embed = nn.Embedding(self.dictionary.size, self.dim).to(
            device)  # A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.embed.weight.data.copy_(embeddings)  # put data in tensor
        self.embed.weight.requires_grad = not self.freeze

    def forward(self, inputs):
        return self.dropout(self.embed(inputs))


class TextFieldEmbedderCharacters(nn.Module):

    def __init__(self, dictionaries, config):
        super(TextFieldEmbedderCharacters, self).__init__()
        self.embedder = TextFieldEmbedderTokens(dictionaries, config['embedder'])
        self.padding = self.embedder.dictionary.lookup('PADDING')
        self.seq2vec = CNNMaxpool(self.embedder.dim, config['encoder'])
        self.dropout = nn.Dropout(config['dropout'])
        self.dim_output = self.seq2vec.dim_output
        self.min_word_len = config['min_word_len']

    def forward(self, characters):
        char_vec = self.embedder(characters)
        char_vec = self.seq2vec(char_vec)
        return self.dropout(torch.relu(char_vec))
