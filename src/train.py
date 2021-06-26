import argparse
import json
import os
import time
from pathlib import Path

import torch
from tensorboard_logger import Logger as TBLogger
from torch.utils.data import DataLoader

from data.data_reader import create_datasets
from data.dictionary import create_dictionaries
from training import settings
from training.lrschedule import create_lr_scheduler


def create_model(config, dictionaries):
    from models import model_create
    model = model_create(config['model'], dictionaries)

    print("Model:", model)

    regularization = config['optimizer']['regularization'] if 'regularization' in config['optimizer'] else {}

    print("Parameters:")
    parameters = []
    num_params = 0
    for key, value in dict(model.named_parameters()).items():
        # 		print(key)
        if not value.requires_grad:
            print("skip ", key)
            continue
        else:
            if key in regularization:
                print("param {} size={} l2={}".format(key, value.numel(), regularization[key]))
                parameters += [{"params": value, "weight_decay": regularization[key]}]
            else:
                print("param {} size={}".format(key, value.numel()))
                parameters += [{"params": value}]
        num_params += value.numel()
    print("total number of params: {} = {}M".format(num_params, num_params / 1024 / 1024 * 4))
    print()

    init_cfg = config['optimizer']['initializer'] if 'initializer' in config['optimizer'] else {}

    print("Initializaing parameters")
    for key, param in dict(model.named_parameters()).items():
        for initializer in [y for x, y in init_cfg.items() if x in key]:
            if initializer == 'orthogonal':
                # is this correct for RNNs, don't think so ?
                print("ORTHOGONAL", key, param.data.size())
                torch.nn.init.orthogonal_(param.data)
            elif initializer == 'rnn-orthogonal':
                print("before:", param.data.size(), param.data.sum().item())
                for tmp in torch.split(param.data, param.data.size(1), dim=0):
                    torch.nn.init.orthogonal_(tmp)
                    print("RNN-ORTHOGONAL", key, tmp.size(), param.data.sum().item())
                print("after:", param.data.size(), param.data.sum().item())
            elif initializer == 'xavier_normal':
                before = param.data.norm().item()
                torch.nn.init.xavier_normal_(param.data)
                after = param.data.norm().item()
                print("XAVIER_NORMAL", key, param.data.size(), before, "->", after)
            break

    print()

    return model, parameters


def do_evaluate(model, dataset, metrics, batch_size, filename=None, tb_logger=None, epoch=-1):
    device = torch.device(settings.device)
    collate_fn = model.collate_func(datasets, device)

    loader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    name = dataset.name

    model.eval()

    for m in metrics:
        m.step()

    if hasattr(model, 'begin_epoch'):
        model.begin_epoch()

    if filename is None:
        total_obj = 0
        for i, minibatch in enumerate(loader):
            obj, outputs = model.forward(**minibatch, metrics=metrics)
            total_obj += obj.item()
    else:
        print("Writing predictions to {}".format(filename))
        with open(filename, 'w') as file:
            total_obj = 0
            for i, minibatch in enumerate(loader):
                obj, predictions = model.predict(**minibatch, metrics=metrics)
                total_obj += obj.item()

                for pred in predictions:
                    json.dump(pred, file)
                    file.write('\n')

    tb_logger.log_value('{}/loss'.format(name), total_obj, epoch)

    for m in metrics:
        m.print(name, True)
        m.log(tb_logger, name)

    print("{}-avg-loss: {}".format(name, total_obj / len(loader)))

    if hasattr(model, 'end_epoch'):
        model.end_epoch(name)


def train(model, datasets, config):
    # device = torch.device("cuda")
    device = torch.device(settings.device)
    model = model.to(device)

    collate_fn = model.collate_func(datasets, device)
    max_epochs = config['optimizer']['iters']
    batch_size = config['optimizer']['batch_size']
    # lrate0 = config['optimizer']['lrate0']

    filename = config['optimizer'].get('init-model', None)
    if filename is not None:
        print("Initializing model {}".format(filename))
        model.load_model(filename, config['model'])

    train = DataLoader(datasets[config['trainer']['train']], collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

    opt_type = config['optimizer'].get('optimizer', 'adam')
    swa = False
    if opt_type == 'adam':
        weight_decay = config['optimizer'].get('weight_decay', 0.0)
        print("ADAM: weight_decay={}".format(weight_decay))
        optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay)  # , lr=lrate0
    elif opt_type == 'asgd':
        t0 = config['optimizer']['t0'] * len(train)
        print("asgd t0:", t0)
        optimizer = torch.optim.ASGD(parameters, t0=t0)
    elif opt_type == 'adam-swa':
        import torchcontrib.optim
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        optimizer = torchcontrib.optim.SWA(optimizer)
        swa_start = config['optimizer']['swa-start']
        swa_freq = config['optimizer']['swa-freq']
        swa = True

    # setup logger
    tb_logger = TBLogger(config['path'])

    print("Start optimization for {} iterations with batch_size={}".format(max_epochs, batch_size))
    name2progress = {name: {'name': name} for name in config['trainer']['evaluate']}

    scheduler = create_lr_scheduler(optimizer, config, max_epochs, len(train))

    # TODO: move this to lrschedule.py
    monitor = None
    factor = 1.0
    patience = 0
    if 'lrate-schedule' in config['trainer']:
        # try factor = 0.9
        monitor = name2progress[config['trainer']['lrate-schedule']['monitor']]
        factor = config['trainer']['lrate-schedule']['factor']
        module = config['trainer']['lrate-schedule']['module']
        metric = config['trainer']['lrate-schedule']['metric']
        patience = config['trainer']['lrate-schedule']['patience']

    metrics = {name: model.create_metrics() for name in config['trainer']['evaluate']}

    if hasattr(model, 'init'):
        model.init(train)

    model.tb_logger = tb_logger

    for epoch in range(max_epochs):
        # print('optimizer', optimizer.state)

        # TODO: move this to lrschedule.py
        if 'lrate-schedule' in config['trainer']:
            print(monitor)
            if module in monitor:
                if (monitor[module][metric]['stall'] + 1) % patience == 0:
                    print("decrease lrate")
                    lrate *= factor
                    for group in optimizer.param_groups:
                        group['lr'] = lrate
            else:
                print("WARNING: no such module:", module)

        model.train()
        total_obj = 0
        tic = time.time()
        max_norm = 0

        if hasattr(model, 'begin_epoch'):
            model.begin_epoch()

        # train
        for i, minibatch in enumerate(train):
            # print("BEGIN\t", torch.cuda.memory_allocated() // (1024*1024))

            lrate = scheduler.step()
            obj, _ = model.forward(**minibatch)
            if obj is None or isinstance(obj, int):
                print("SKIP EMPTY MINIBATCH")
                continue
            total_obj += obj.item()

            optimizer.zero_grad()
            obj.backward()
            if 'clip-norm' in config['optimizer']:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip-norm'])
                if norm > max_norm:
                    max_norm = norm

            optimizer.step()

        if hasattr(model, 'end_epoch'):
            model.end_epoch('train')

        if swa and epoch >= swa_start and epoch % swa_freq == 0:
            print("UPDATE-SWA")
            optimizer.update_swa()

        print("{}\tobj: {}   time: {}    lrate: {}".format(epoch, total_obj, time.time() - tic, lrate))

        tb_logger.log_value('train/loss', total_obj, epoch)
        tb_logger.log_value('train/max-norm', max_norm, epoch)
        tb_logger.log_value('optimizer/lrate', lrate, epoch)

        for name, param in model.named_parameters():
            tb_logger.log_value('parameters/mean/{}'.format(name), param.mean().item(), epoch)
            tb_logger.log_value('parameters/stdv/{}'.format(name), param.std().item(), epoch)
            tb_logger.log_value('parameters/norm/{}'.format(name), param.norm().item(), epoch)
            tb_logger.log_value('parameters/max/{}'.format(name), param.max().item(), epoch)

        try:
            if config['optimizer']['write-iter-model']:
                model.write_model('{}/{}.model'.format(config['path'], epoch))
        except:
            print("ERROR: failed to write model to disk")

        if swa and epoch >= swa_start:
            optimizer.swap_swa_sgd()

        # evaluate
        with torch.no_grad():
            for name in config['trainer']['evaluate']:
                predict_file = '{}/{}.json'.format(config['path'], name) if config['trainer'][
                    'write-predictions'] else None
                do_evaluate(model, datasets[name], metrics[name], batch_size, predict_file, tb_logger, epoch)
                # do_evaluate(model, datasets[name], metrics[name], batch_size, predict_file, tb_logger, epoch)

        if swa and epoch >= swa_start:
            optimizer.swap_swa_sgd()

    while True:
        try:
            if config['optimizer']['write-last-model']:
                model.write_model('{}/last.model'.format(config['path']))
                break
        except:
            print("ERROR: failed to write model to disk")
            time.sleep(60)


if __name__ == "__main__":

    print("Start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="configuration file")
    parser.add_argument("--path", dest="path", type=str, default=None)
    parser.add_argument("--device", dest="device", type=str, default="cuda")

    args = parser.parse_args()

    settings.device = args.device

    with open(args.config_file) as f:
        config = json.load(f)

    if 'path' not in config:
        print('set path=', Path(args.config_file).parent)
        config['path'] = Path(args.config_file).parent

    if args.path is not None:
        # print("WARNING: setting path to {}".format(args.path))
        # adds the config file to the path as directory, so the structure is clear
        path = args.path
        config_name = os.path.basename(args.config_file)
        config_name = config_name[:config_name.rindex('.')]
        path = os.path.join(path, config_name)
        os.makedirs(path, exist_ok=True)
        config['path'] = path
        config['model']['path'] = path

    # added: write the config_file for reproducibility
    with open(config['path'] + '/config_file.json', 'w') as temp:
        j = json.dumps(config, indent=4)
        print(j, file=temp)
        temp.close()

    dictionaries = create_dictionaries(config, True)
    datasets, data, evaluate = create_datasets(config, dictionaries)
    model, parameters = create_model(config, dictionaries)
    train(model, datasets, config)
    torch.save(model.state_dict(), config['model']['path'] + '/savemodel.pth')
