class MetricLinkerImproved:

    def __init__(self, task):
        self.task = task
        self.epoch = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def step(self):
        self.epoch += 1
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update2(self, args, metadata={}):
        for pred, gold in zip(args['pred'], args['gold']):
            P = set(pred)
            G = set(gold)

            self.tp += len(P & G)
            self.fp += len(P - G)
            self.fn += len(G - P)

    def get_pr(self):
        return self.tp / (self.tp + self.fp) if self.tp != 0 else 0.0

    def get_re(self):
        return self.tp / (self.tp + self.fn) if self.tp != 0 else 0.0

    def get_f1(self):
        return 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn) if self.tp != 0 else 0.0

    def print(self, dataset_name, details=False):
        print("EVAL-LINKER\t{}-{}\ttp: {}\tfp: {}\tfn: {}\tpr: {}\tre: {}\tf1: {}\n".format(dataset_name, self.task,
                                                                                            self.tp, self.fp, self.fn,
                                                                                            self.get_pr(),
                                                                                            self.get_re(),
                                                                                            self.get_f1()))

    def log(self, tb_logger, dataset_name):
        print("SKIP")
