# coding=utf-8
import torch
import torch.nn as nn

def get_params(alg, args, inner=False, isteacher=False):
    if args.schuse:
        if args.schusech == 'cos':
            init_lr = args.lr
        else:
            init_lr = 1.0
    else:
        if inner:
            init_lr = args.inner_lr
        else:
            init_lr = args.lr
    if isteacher:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
        return params
    if args.algorithm == "Our":
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
    elif args.algorithm == "ERM":
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
    elif args.algorithm == "Fixed":
        params = [
        {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
        {'params': alg.bottleneck.parameters(), 'lr': args.lr_decay2 * init_lr},
        {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr}
    ]
        params.append({'params': alg.discriminator.parameters(),
                        'lr': args.lr_decay2 * init_lr})

    elif args.algorithm == "CORAL":
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
    elif args.algorithm == "RSC":
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
    elif args.algorithm == "DANN":
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
        params.append({'params': alg.discriminator.parameters(),
                    'lr': args.lr_decay2 * init_lr})
    elif args.algorithm == "ANDMask":
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
    elif args.algorithm == "GroupDRO":
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
    elif args.algorithm == "Mixup":
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * init_lr}
        ]
    return params


def get_optimizer(alg, args,inner=False, isteacher=False):
    params = get_params(alg, args,inner, isteacher)
    optimizer = torch.optim.Adam(
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, 0.9))
    return optimizer


def get_scheduler(optimizer, args):
    return None
