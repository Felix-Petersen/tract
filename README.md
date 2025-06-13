# TrAct - Training Activations

Official implementation for our NeurIPS 2024 paper [TrAct: Making First-layer Pre-Activations Trainable](https://arxiv.org/pdf/2410.23970).
In this work, we provide TrAct, a method for more effective and efficient training of the first layer that accelerates training by between 25% and 300%.
Herein, rather than using the training dynamics of training weight, we provide a closed form solution for training activations by indirectly updating weights, leading to faster and better training.

Video @ [YouTube](https://www.youtube.com/watch?v=ZjTAjjxbkRY).

## 💻 Installation

TrAct can be installed via pip from PyPI with

```shell
pip install torch_tract
```

Alternatively, it is sufficient to copy the `src/tract.py` into your existing project.

## 👩‍💻 Usage

The `tract.py` file contains the `TrAct` wrapper, which replaces `torch.nn.Linear` and `torch.nn.Conv2d` modules by `TrActLinear` and `TrActConv2d` modules, respectively.

After initialization, simply wrap your first layer in a `TrAct(layer, l=l)`, wherein `l` / λ is the only hyperparameter.
The default for this is `l=0.1` and performs very well. 
Please refer to the [paper](https://arxiv.org/pdf/2410.23970) for additional information on the hyperparameter.

In the script, `TrAct` is implemented via:

```python
# regular initialization of model
model = resnet18(num_classes=num_classes)
# apply TrAct to first layer
model.conv1 = TrAct(model.conv1, l=args.l)
```

Using the same change, it can also be applied to, e.g., ResNet training on ImageNet.
Analogously, it can be applied for other codebases and experiments, e.g.:

```python
# For CIFAR ViT: https://github.com/omihub777/ViT-CIFAR/
model.emb = TrAct(model.emb, l=args.l)
# For DeiT: https://github.com/facebookresearch/deit/
model.patch_embed.proj = TrAct(model.patch_embed.proj, l=args.l)
```


<details>
  <summary>To reproduce the first seed of Figure 1, run:</summary>

```shell
python train_cifar.py --method normal        --n_epochs 100 --lr 0.08 --optim sgd_w_momentum_cosine --seed 0
python train_cifar.py --method normal        --n_epochs 200 --lr 0.08 --optim sgd_w_momentum_cosine --seed 0
python train_cifar.py --method normal        --n_epochs 400 --lr 0.08 --optim sgd_w_momentum_cosine --seed 0
python train_cifar.py --method normal        --n_epochs 800 --lr 0.08 --optim sgd_w_momentum_cosine --seed 0
python train_cifar.py --method tract --l 0.1 --n_epochs 100 --lr 0.08 --optim sgd_w_momentum_cosine --seed 0
python train_cifar.py --method tract --l 0.1 --n_epochs 200 --lr 0.08 --optim sgd_w_momentum_cosine --seed 0
python train_cifar.py --method tract --l 0.1 --n_epochs 400 --lr 0.08 --optim sgd_w_momentum_cosine --seed 0
python train_cifar.py --method tract --l 0.1 --n_epochs 800 --lr 0.08 --optim sgd_w_momentum_cosine --seed 0

python train_cifar.py --method normal        --n_epochs 100 --lr 0.010 --optim adam_cosine --seed 0
python train_cifar.py --method normal        --n_epochs 200 --lr 0.010 --optim adam_cosine --seed 0
python train_cifar.py --method normal        --n_epochs 400 --lr 0.001 --optim adam_cosine --seed 0
python train_cifar.py --method normal        --n_epochs 800 --lr 0.001 --optim adam_cosine --seed 0
python train_cifar.py --method tract --l 0.1 --n_epochs 100 --lr 0.010 --optim adam_cosine --seed 0
python train_cifar.py --method tract --l 0.1 --n_epochs 200 --lr 0.010 --optim adam_cosine --seed 0
python train_cifar.py --method tract --l 0.1 --n_epochs 400 --lr 0.001 --optim adam_cosine --seed 0
python train_cifar.py --method tract --l 0.1 --n_epochs 800 --lr 0.001 --optim adam_cosine --seed 0
```

</details>


## 📖 Citing

```bibtex
@inproceedings{petersen2024tract,
  title={TrAct: Making First-layer Pre-Activations Trainable},
  author={Petersen, Felix and Borgelt, Christian and Ermon, Stefano},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

## License

TrAct is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.

