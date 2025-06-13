import torch
from torchvision import datasets, transforms
import argparse
import random
from tqdm import trange
import numpy as np
if __name__ == '__main__':
    torch.set_num_threads(min(8, torch.get_num_threads()))
from torch_tract import TrAct


if __name__ == '__main__':

    #####################################################

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--optim', default='sgd_w_momentum_cosine', type=str, choices=['adam', 'adam_cosine', 'sgd_w_momentum_cosine'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-m', '--method', type=str, required=True)
    parser.add_argument('--l', type=float, default=0.1)
    parser.add_argument('--no_data_aug', action='store_true')
    parser.add_argument('--no_cudnn', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(vars(args))

    if args.no_cudnn:
        torch.backends.cudnn.enabled = False

    device = torch.device(args.device)

    #####################################################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.no_data_aug:
        transform_train = transform_test

    if args.dataset == 'cifar10':
        data_class = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        data_class = datasets.CIFAR100
        num_classes = 100
    else:
        raise NotImplementedError(args.dataset)

    trainset = data_class(
        root='./data', train=True, download=True, transform=transform_train)

    testset = data_class(
        root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.method == 'normal':
        from resnet_cifar10 import resnet18
        model = resnet18(num_classes=num_classes).to(device)
    elif args.method == 'normal_34':
        from resnet_cifar10 import resnet34
        model = resnet34(num_classes=num_classes).to(device)
    elif args.method == 'normal_50':
        from resnet_cifar10 import resnet50
        model = resnet50(num_classes=num_classes).to(device)
    elif args.method == 'tract':
        from resnet_cifar10 import resnet18
        model = resnet18(num_classes=num_classes).to(device)
        model.conv1 = TrAct(model.conv1, l=args.l)
    elif args.method == 'tract_34':
        from resnet_cifar10 import resnet34
        model = resnet34(num_classes=num_classes).to(device)
        model.conv1 = TrAct(model.conv1, l=args.l)
    elif args.method == 'tract_50':
        from resnet_cifar10 import resnet50
        model = resnet50(num_classes=num_classes).to(device)
        model.conv1 = TrAct(model.conv1, l=args.l)
    else:
        raise NotImplementedError(args.method)
    print(model)

    if args.optim == 'adam':
        optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        scheduler = None
    elif args.optim == 'adam_cosine':
        optim = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs*len(train_loader))
    elif args.optim == 'sgd_w_momentum_cosine':
        optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs*len(train_loader))
    else:
        raise NotImplementedError(args.optim)

    best_v_top1 = 0

    for epoch in trange(args.n_epochs):
        model.train()

        train_acc = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            y_hat = model(x)

            loss = torch.nn.CrossEntropyLoss()(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()

            train_acc.append((y_hat.argmax(1) == y).float().mean().item())

        train_acc = np.mean(train_acc)

        model.eval()

        # TEST
        test_acc = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                test_acc.append((y_hat.argmax(1) == y).float().mean().item())
        test_acc = np.mean(test_acc)

        print('{}:  Train: {:.3f},  Test: {:.3f},  Method: {}'.format(epoch, train_acc, test_acc, args.method))
