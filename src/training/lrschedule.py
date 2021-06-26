def create_lr_scheduler(optimizer, config, max_epochs, num_training_instances):
    if 'lr-scheduler' not in config:
        return MyNoneScheduler(optimizer)
    elif config['lr-scheduler']['type'] == 'linear-decay':
        return MyLinearDecayScheduler(optimizer, config['lr-scheduler'], max_epochs, num_training_instances)
    else:
        raise BaseException("no such scheduler:", config['lr-scheduler']['type'])


class MyLinearDecayScheduler:

    def __init__(self, optimizer, config, num_epoch, steps_per_epoch=1):
        self.optimizer = optimizer
        self.lrate0 = config['lrate0']
        self.gamma = config['gamma']
        self.t0 = config['t0'] * steps_per_epoch
        self.t1 = config['t1'] * steps_per_epoch
        self.t = 1
        self.lrate = 0

    def step(self):
        self.t += 1
        if self.t <= self.t0:
            self.lrate = self.lrate0
        elif self.t <= self.t1:
            fraction = (self.t - self.t0) / (self.t1 - self.t0)
            self.lrate = self.lrate0 * (self.gamma * fraction + 1.0 * (1 - fraction))

        for group in self.optimizer.param_groups:
            group['lr'] = self.lrate

        return self.lrate


class MyNoneScheduler:

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        for group in self.optimizer.param_groups:
            return group['lr']
