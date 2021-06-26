from data.predictions_serializer import convert_to_json
from models.coref_loss import LossCoref
from models.coref_scorer import ModuleCorefScorer
from models.embedders.kb_embedder import *
from models.embedders.span_embedder import SpanEndpoint
from models.embedders.text_embedder import TextEmbedder
from models.misc.collate import main_collate
from models.misc.misc import SpanPairs, Seq2Seq, create_all_spans, span_intersection
from models.ner import TaskNER
from models.pruner import MentionPruner
from models.re_loss import LossRelationsX
from models.re_scorer import ModuleRelScorer
from training import settings


def create_task_relations(name, config, labels):
    if config['type'] == 'binary-x':
        return LossRelationsX(name, config, labels)
    else:
        raise BaseException("no such relation task:", config['type'])


def create_kbentities(model, dictionaries, config):
    # if 'kb_embedder' in config:
    # model.span_extractor.dim_output += config['kb_embedder']['dim']
    return KBEmbedderAll(dictionaries, config['kb_embedder'], model.span_extractor.dim_output)
    # return None


def create_corefprop(model, config, dim_input):
    cp_type = config['corefprop']['type']

    if cp_type == 'none':
        return None
    elif cp_type == 'default':
        return ModuleCorefScorer(dim_input, model.span_pruner.scorer, model.span_pair_generator, config['corefprop'])
        # return ModuleCorefScorer(model.span_extractor.dim_output, model.span_pruner.scorer, model.span_pair_generator,
        #                          config['corefprop'])
    else:
        raise BaseException("no such corefprop:", cp_type)


def create_relprop(model, config, dim_input):
    rp_type = config['relprop']['type']

    if rp_type == 'none':
        return None
    elif rp_type == 'default':
        return ModuleRelScorer(dim_input, model.span_pair_generator, model.relation_labels,
                               config['relprop'])
    else:
        raise BaseException("no such relprop:", rp_type)


class MainModel(nn.Module):  # inherited from module, will describe the layers of the model (can be nested with modules)

    def __init__(self, dictionaries, config):
        super(MainModel, self).__init__()
        # self.random_embed_dim = config['random_embed_dim']
        self.max_span_length = config['max_span_length']
        self.hidden_dim = config['hidden_dim']  # 150
        self.hidden_dp = config['hidden_dropout']  # 0.4
        # self.rel_after_coref = config['rel_after_coref']
        self.debug_memory = False
        self.counter = -1

        self.embedder = TextEmbedder(dictionaries, config['text_embedder'])
        self.emb_dropout = nn.Dropout(config['lexical_dropout'])
        self.seq2seq = Seq2Seq(self.embedder.dim_output, config['seq2seq'])
        self.span_extractor = SpanEndpoint(self.seq2seq.dim_output, self.max_span_length, config['span-extractor'])

        # add entities to all spans
        # self.kb_entities = create_kbentities_all(self, dictionaries, config)
        self.kb_entities: KBEmbedderAll = create_kbentities(self, dictionaries, config)

        self.span_pruner = MentionPruner(self.kb_entities.all_dim, self.max_span_length, config['pruner'])
        # add entities to the pruned spans

        self.span_pair_generator = SpanPairs(self.kb_entities.filtered_dim, config['span-pairs'])

        self.coref_scorer = create_corefprop(self, config, self.kb_entities.filtered_dim)

        self.relation_labels = dictionaries['relations'].tolist()
        self.rel_scorer = create_relprop(self, config, self.kb_entities.filtered_dim)

        self.coref_task = LossCoref('coref', config['coref'])

        self.ner_task = TaskNER('tags', self.kb_entities.filtered_dim, dictionaries['tags-y'], config['ner'])
        self.relation_task = create_task_relations('rels', config['relations'], self.relation_labels)

        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_denom = 0
        self.num = 0

        if not self.span_pruner.sort_after_pruning and self.pairs.requires_sorted_spans:
            raise BaseException("ERROR: spans MUST be sorted")

    def collate_func(self, datasets, device):
        return lambda x: main_collate(self, x, device)

    def end_epoch(self, dataset_name):
        print("{}-span-generator: {} / {} = {}".format(dataset_name, self.span_generated, self.span_recall_denom,
                                                       self.span_generated / self.span_recall_denom))
        print("{}-span-recall: {} / {} = {}".format(dataset_name, self.span_recall_numer, self.span_recall_denom,
                                                    self.span_recall_numer / self.span_recall_denom))
        self.span_generated = 0
        self.span_recall_numer = 0
        self.span_recall_denom = 0

    def forward(self, input_batch, metrics=[]):

        output = {}  # output om the model
        self.counter = self.counter + 1
        inputs = input_batch['inputs']
        metadata = input_batch['metadata']
        sequence_lengths = inputs['sequence_lengths']

        if self.debug_memory:
            print("START", sequence_lengths)
            print("(none)  ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        # MODEL MODULES
        embeddings = self.embedder(inputs['characters'], inputs['tokens'])

        embeddings = self.emb_dropout(embeddings)

        hidden = self.seq2seq(embeddings, sequence_lengths, inputs['token_indices']).contiguous()

        # create feature of candidates;
        # TODO: verify!, all the logic should be inside the forward(), not apart in add_weighted

        # create span
        # create all possible spans of length 5
        span_begin, span_end = create_all_spans(hidden.size(0), hidden.size(1), self.max_span_length)
        span_begin, span_end = span_begin.to(settings.device), span_end.to(settings.device)

        # here assert to compare the nr of spans used for candidates (extracted in data_reader.py) vs the nr of spans
        # calculated here
        assert span_begin.shape[-1] * span_begin.shape[-2] == inputs['kb_entities']['len_candidates'].shape[-1]
        span_mask = (span_end < sequence_lengths.unsqueeze(-1).unsqueeze(-1)).float().to(settings.device)

        # extract span embeddings
        span_vecs = self.span_extractor(hidden, span_begin, span_end, self.max_span_length)

        all_spans = {
            'span_vecs': span_vecs,
            'span_begin': span_begin,
            'span_end': span_end,
            'span_mask': span_mask
        }

        if self.kb_entities.spans_scope == 'all':
            all_spans['span_vecs'] = self.kb_entities(None, all_spans, inputs)

        # prune spans
        all_spans, filtered_spans = self.span_pruner(all_spans, sequence_lengths)
        pred_spans = filtered_spans['spans']
        gold_spans = inputs['gold_spans']

        # TODO: tensorize
        self.span_generated += sum([len(x) for x in pred_spans])
        self.span_recall_numer += span_intersection(pred_spans, gold_spans)
        self.span_recall_denom += sum([len(x) for x in gold_spans])

        if self.kb_entities.spans_scope == 'filtered':
            filtered_spans['span_vecs'], all_spans['span_vecs'] = self.kb_entities(filtered_spans, all_spans, inputs)

        if self.debug_memory:
            print("(pruner)", torch.cuda.memory_allocated(0) / 1024 / 1024)

        ## coref
        if self.coref_task.enabled:
            coref_all, coref_filtered, coref_scores = self.coref_scorer(all_spans, filtered_spans, sequence_lengths)
        else:
            coref_all = all_spans
            coref_filtered = filtered_spans
            coref_scores = None

        if self.debug_memory:
            print("(coref) ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        ## relations
        if self.relation_task.enabled:
            relation_all, relation_filtered, relation_scores = self.rel_scorer(coref_all, coref_filtered)

        else:
            relation_all = coref_all
            relation_filtered = coref_filtered
            relation_scores = None

        if self.debug_memory:
            print("(rels)  ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        # LOSS FUNCTIONS

        ## coref
        coref_obj, output['coref'] = self.coref_task(coref_scores, gold_m2i=inputs['gold_m2i'], pred_spans=pred_spans,
                                                     gold_spans=gold_spans,
                                                     # predict=not self.training
                                                     predict=True)

        ## ner
        ner_obj, output['tags'] = self.ner_task(
            relation_all,
            sequence_lengths,
            inputs['gold_tags_indices']
        )

        ## relations
        rel_obj, output['rels'] = self.relation_task(
            relation_filtered,
            relation_scores,
            inputs,
            output['coref'],
            predict=not self.training
        )

        for m in metrics:
            if m.task in output:
                m.update2(output[m.task], metadata)

        if self.debug_memory:
            print("(loss)  ", torch.cuda.memory_allocated(0) / 1024 / 1024)

        return coref_obj + ner_obj + rel_obj, output

    # def forward(self, input_batch, metrics=[]):

    def predict(self, input_batch, metrics=[]):
        loss, output = self.forward(input_batch, metrics)
        return loss, self.decode(input_batch['metadata'], output)

    def create_metrics(self):
        return self.coref_task.create_metrics() + self.ner_task.create_metrics() + self.relation_task.create_metrics()

    def write_model(self, filename):
        return

    def load_model(self, filename, config, to_cpu=False):
        return

    def decode(self, metadata, outputs):
        predictions = []

        if outputs['coref']['pred'] is None:
            outputs['coref']['pred'] = [None for _ in metadata['identifiers']]
            outputs['coref']['gold'] = [None for _ in metadata['identifiers']]

        if outputs['rels']['pred'] is None:
            outputs['rels']['pred'] = [None for _ in metadata['identifiers']]
            outputs['rels']['gold'] = [None for _ in metadata['identifiers']]

        for identifier, content, begin, end, ner, coref, rels in zip(metadata['identifiers'], metadata['content'],
                                                                     metadata['begin'], metadata['end'],
                                                                     outputs['tags']['pred'], outputs['coref']['pred'],
                                                                     outputs['rels']['pred']):
            predictions.append(convert_to_json(identifier, content, begin.tolist(), end.tolist(), ner, coref, rels))

        return predictions
