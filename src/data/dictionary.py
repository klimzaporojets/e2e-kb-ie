import json


def load_dictionary(config, path):
    type = config['type']
    filename = config['filename']
    filename = filename if filename.startswith("/") else "{}/{}".format(path, filename)

    if type == 'json':
        dictionary = Dictionary()
        dictionary.load_json(filename)
    else:
        raise BaseException("no such type", type)

    return dictionary


def create_dictionaries(config, training):
    path = config['path']

    print("Loading dictionaries (training={})".format(training))

    if 'dictionaries' in config:
        dictionaries = {}
        for name, dict_config in config['dictionaries'].items():
            if training:
                if "init" in dict_config:
                    dictionary = load_dictionary(dict_config['init'], path)
                    # print('init {}: size={}'.format(name, dictionary.size))
                else:
                    # print("init {} (blank)".format(name))
                    dictionary = Dictionary()
            else:
                dictionary = load_dictionary(dict_config, path)
                print('load {}: size={}'.format(name, dictionary.size))

            dictionary.prefix = dict_config['prefix'] if 'prefix' in dict_config else ''

            if 'rewriter' in dict_config:
                if dict_config['rewriter'] == 'lowercase':
                    dictionary.rewriter = lambda t: t.lower()
                elif dict_config['rewriter'] == 'none':
                    print("rewriter: none")
                else:
                    raise BaseException("no such rewriter", dict_config['rewriter'])

            if 'append' in dict_config:
                for x in dict_config['append']:
                    idx = dictionary.add(x)
                    print("   add token", x, "->", idx)

            if 'unknown' in dict_config:
                dictionary.set_unknown_token(dict_config['unknown'])

            if 'debug' in dict_config:
                dictionary.debug = dict_config['debug']

            if 'update' in dict_config:
                dictionary.update = dict_config['update']

            if not training:
                dictionary.update = False

            dictionaries[name] = dictionary

        return dictionaries
    else:
        print("WARNING: using wikipedia dictionary")
        words = Dictionary()
        entities = Dictionary()

        words.set_unknown_token("UNKNOWN")
        words.load_spirit_dictionary('data/tokens.dict', 5)
        entities.set_unknown_token("UNKNOWN")
        entities.load_spirit_dictionary('data/entities.dict', 5)
        return {
            'words': words,
            'entities': entities
        }


class Dictionary:

    def __init__(self):
        self.rewriter = lambda t: t
        self.debug = False
        self.token_unknown = -1
        self.update = True
        self.prefix = ''
        self.tmp_unknown = None

        self.clear()

    def clear(self):
        self.word2idx = {}
        self.matrix = False
        self.size = 0
        self.out_of_voc = 0
        self.oov = set()

        if self.tmp_unknown is not None:
            self.token_unknown = self.lookup(self.tmp_unknown)

    def load_json(self, filename):
        with open(filename) as file:
            data = json.load(file)
            if isinstance(data, (list,)):
                for idx, word in enumerate(data):
                    if self.lookup(word) != idx:
                        print("WARNING: invalid dictionary")
            else:
                for word, idx in data.items():
                    if self.lookup(word) != idx:
                        print("WARNING: invalid dictionary")

    def lookup(self, token):
        token = self.prefix + self.rewriter(token)
        if not token in self.word2idx:
            if self.update:
                self.word2idx[token] = self.size
                self.size += 1
            else:
                if self.debug:
                    print("oov: '{}' -> {}".format(token, self.token_unknown))
                self.out_of_voc += 1
                return self.token_unknown
        return self.word2idx[token]

    def add(self, token):
        if not token in self.word2idx:
            self.word2idx[token] = self.size
            self.size += 1
        return self.word2idx[token]

    def set_unknown_token(self, unknown_token):
        self.tmp_unknown = unknown_token
        self.token_unknown = self.word2idx[self.prefix + unknown_token]
        print(self.get(self.token_unknown), "->", self.token_unknown)

    def write(self, filename):
        import json
        with open(filename, 'w') as file:
            json.dump(self.word2idx, file)

    def get(self, index):
        for word, idx in self.word2idx.items():
            if idx == index:
                return word
        return None

    def tolist(self):
        list = [None] * self.size
        for word, idx in self.word2idx.items():
            list[idx] = word
        return list
