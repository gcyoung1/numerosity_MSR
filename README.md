# Meta-learning Symmetries by Reparameterization (MSR)
Code and weights corresponding to Meta-learning Symmetries by Reparameterization, found at this [arXiv link](https://arxiv.org/abs/2007.02933).

## Installation and requirements
All experiments were run in a conda environment with python 3.7.5, using pytorch. The conda environment we used is exported in `environment.yml`, though it likely contains more packages than are strictly necessary.

## Synthetic experiments

### Generate data
The file `generate_synthetic_data.py` contains the data generation code.
You must specify the name of the problem we wish to generate data for. Options are:
* `rank1`: 1-D data generated by a convolution layer
* `rank2`: 1-D data generated by a locally connected layer, with rank 2 weight factorization.
* `rank5`: 1-D data generated by a locally connected layer, with rank 5 weight factorization.
* `2d_rot8`: 2-D data that is equivariant to 45-degree rotations.
* `2d_rot8_flip`: 2-D data that is equivariant to 45-degree rotations and flips.

Example:
```sh
python generate_synthetic_data.py --problem rank1
```

### Train
The file `train_synthetic.py` contains training and evaluation code. Specifying the argument `model` controls whether to use MSR, plain MAML, or something else.

Some examples are given below:
```sh
python train_synthetic.py --problem rank1 --model share_fc  # MSR+FC on rank1 problem.
python train_synthetic.py --problem rank1 --model fc  # MAML+FC on rank1 problem.
python train_synthetic.py --problem rank1 --model conv  # MAML+Conv model on rank1 problem.
python train_synthetic.py --problem 2d_rot8 --model share_conv  # MSR+Conv on 2d_rot8 problem.
python train_synthetic.py --problem 2d_rot8 --model conv  # MAML+Conv on 2d_rot8 problem.
```

## Augmented-(Omniglot/Miniimagenet)

WIP
