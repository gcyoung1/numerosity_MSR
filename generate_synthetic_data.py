"""Generate synthetic data for experiments."""

import argparse
import os
#from e2cnn import gspaces
#from e2cnn import nn as gnn
from scipy.special import softmax
import skimage
import scipy.ndimage
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from layers import LocallyConnected1d


def generate_1d(out_path):
    lc_layer = LocallyConnected1d(1, 1, 68, bias=False)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt = np.random.randn(1, 1, 1, 1, 3).astype(np.float32)
        filt = np.repeat(filt, 68, axis=3)
        ws.append(filt)
        lc_layer.weight = nn.Parameter(torch.from_numpy(filt))
        task_xs, task_ys = [], []
        inp = np.random.randn(20, 1, 70).astype(np.float32)
        result = lc_layer(torch.from_numpy(inp))  # (20, 1, 68)
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)

def generate_1d_stack(out_path):
    lc_layer_1 = LocallyConnected1d(1, 1, 68, bias=False)
    lc_layer_2 = LocallyConnected1d(1, 1, 66, bias=False)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        filt1 = np.random.randn(1, 1, 1, 1, 3).astype(np.float32)
        filt1 = np.repeat(filt1, 68, axis=3)
        filt2 = np.random.randn(1, 1, 1, 1, 3).astype(np.float32)
        filt2 = np.repeat(filt2, 66, axis=3)
        lc_layer_1.weight = nn.Parameter(torch.from_numpy(filt1))
        lc_layer_2.weight = nn.Parameter(torch.from_numpy(filt2))
        task_xs, task_ys = [], []
        inp = np.random.randn(20, 1, 70).astype(np.float32)
        result = lc_layer_1(torch.from_numpy(inp))  # (20, 1, 68)
        result = lc_layer_2(F.relu(result))  # (20, 1, 64)
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    ws = np.zeros_like(ys)
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)


def generate_1d_low_rank(out_path, rank=2):
    lc_layer = LocallyConnected1d(1, 1, 68, bias=False)
    xs, ys, ws = [], [], []
    connectivity = softmax(np.random.randn(68, rank), axis=1)  # shape == (68, rank)
    for task_idx in range(10000):
        basis = np.random.randn(rank, 3)
        filt = np.dot(connectivity, basis)  # shape == (68, 3)
        filt = np.reshape(filt, (1, 1, 1, 68, 3)).astype(np.float32)
        ws.append(filt)
        lc_layer.weight = nn.Parameter(torch.from_numpy(filt))
        task_xs, task_ys = [], []
        inp = np.random.randn(20, 1, 70).astype(np.float32)
        result = lc_layer(torch.from_numpy(inp))  # (20, 1, 68)
        result = result.cpu().detach().numpy()
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)


def generate_2d_rot8(out_path):
    r2_act = gspaces.Rot2dOnR2(N=8)
    feat_type_in = gnn.FieldType(r2_act, [r2_act.trivial_repr])
    feat_type_out = gnn.FieldType(r2_act, 3 * [r2_act.regular_repr])
    conv = gnn.R2Conv(feat_type_in, feat_type_out, kernel_size=3, bias=False)
    xs, ys, ws = [], [], []
    for task_idx in range(10000):
        gnn.init.generalized_he_init(conv.weights, conv.basisexpansion)
        inp = gnn.GeometricTensor(torch.randn(20, 1, 32, 32), feat_type_in)
        result = conv(inp).tensor.detach().cpu().numpy()
        xs.append(inp.tensor.detach().cpu().numpy())
        ys.append(result)
        ws.append(conv.weights.detach().cpu().numpy())
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)


def generate_2d_rot8_flip(out_path):
    r2_act = gspaces.FlipRot2dOnR2(N=8)
    feat_type_in = gnn.FieldType(r2_act, [r2_act.trivial_repr])
    feat_type_out = gnn.FieldType(r2_act, 3 * [r2_act.regular_repr])
    xs, ys, ws = [], [], []
    device = torch.device("cuda")
    conv = gnn.R2Conv(feat_type_in, feat_type_out, kernel_size=3, bias=False).to(device)
    for task_idx in range(2000):
        gnn.init.generalized_he_init(conv.weights, conv.basisexpansion)
        inp = gnn.GeometricTensor(torch.randn(20, 1, 32, 32).to(device), feat_type_in)
        result = conv(inp).tensor.detach().cpu().numpy()
        xs.append(inp.tensor.detach().cpu().numpy())
        ys.append(result)
        ws.append(conv.weights.detach().cpu().numpy())
        del inp, result
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys, ws = np.stack(xs), np.stack(ys), np.stack(ws)
    np.savez(out_path, x=xs, y=ys, w=ws)


def generate_1d_image(structure, num_ccs, should_match, length):
    while True:
        inp = np.random.randint(low=0, high=2, size=(1, length)).astype(np.float32)
        matches = scipy.ndimage.label(inp[0,:], structure)[1] == num_ccs
        if matches and should_match:
            return inp
        if not matches and not should_match:
            return inp

def generate_2d_image(structure, num_ccs, should_match):
    while True:
        inp = np.random.randint(low=0, high=2, size=(1, 32, 32)).astype(np.float32)
        matches = scipy.ndimage.label(inp[0,:,:], structure)[1] == num_ccs
        if matches and should_match:
            return inp
        if not matches and not should_match:
            return inp

def generate_1d_cc_contrastive(out_path, length):
    structure = np.array([1,1,1])
    xs, ys = [], []
    num_ccs_dict = {}
    for task_idx in range(10000):
        result = []
        num_ccs = -1
        inp = []
        for i in range(20):
            if num_ccs == -1:
                new_inp = np.random.randint(low=0, high=2, size=(1, length)).astype(np.float32)
                num_ccs = scipy.ndimage.label(new_inp[0,:], structure)[1]
                num_ccs_dict[num_ccs] = num_ccs_dict.get(num_ccs, 0) + 1
            else:
                new_inp = generate_1d_image(structure, num_ccs, should_match=True, length=length)
            inp.append(new_inp)
            pos_inp = generate_1d_image(structure, num_ccs, should_match=True, length=length)
            neg_inp = generate_1d_image(structure, num_ccs, should_match=False, length=length)
            result.append(np.concatenate([pos_inp, neg_inp]))
        xs.append(np.array(inp))
        ys.append(np.array(result))
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys = np.stack(xs), np.stack(ys)
    ws = np.zeros_like(ys)
    np.savez(out_path, x=xs, y=ys, w=ws)
    print("".join([f"Examples with {k} ccs: {v}\n" for k, v in num_ccs_dict.items()]))

def generate_2d_cc_contrastive(out_path, connectivity = 1):
    assert connectivity in [1,2]
    if connectivity == 1:
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]])
    elif connectivity == 2:
        structure = np.array([[1,1,1],
                              [1,1,1],
                              [1,1,1]])

    xs, ys = [], []
    for task_idx in range(10000):
        inp = []
        num_ccs = -1
        result = []
        num_ccs_dict = {}
        for i in range(20):
            if num_ccs == -1:
                new_inp = np.random.randint(low=0, high=2, size=(1, 32, 32)).astype(np.float32)
                num_ccs = scipy.ndimage.label(new_inp[0,:,:], structure)[1]
                num_ccs_dict[num_ccs] = num_ccs_dict.get(num_ccs, 0) + 1
            else:
                new_inp = generate_2d_image(structure, num_ccs, should_match=True)
            inp.append(new_inp)
            pos_inp = generate_2d_image(structure, num_ccs, should_match=True)
            neg_inp = generate_2d_image(structure, num_ccs, should_match=False)
            result.append(np.concatenate([pos_inp, neg_inp]))
        xs.append(np.array(inp))
        ys.append(np.array(result))
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys = np.stack(xs), np.stack(ys)
    ws = np.zeros_like(ys)
    np.savez(out_path, x=xs, y=ys, w=ws)
    print("".join([f"Examples with {k} ccs: {v}\n" for k, v in num_ccs_dict.items()]))

def generate_1d_cc(out_path, length):
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]])
    xs, ys = [], []
    for task_idx in range(10000):
        inp = np.random.randint(low=0, high=2, size=(20, 1, length)).astype(np.float32)
        result = np.expand_dims(np.array([scipy.ndimage.label(vec, structure)[1] for vec in inp]), (1,2))
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys = np.stack(xs), np.stack(ys)
    ws = np.zeros_like(ys)
    np.savez(out_path, x=xs, y=ys, w=ws)

def generate_2d_cc(out_path, connectivity = 1):
    assert connectivity in [1,2]
    if connectivity == 1:
        structure = np.array([[0,1,0],
                              [1,1,1],
                              [0,1,0]])
    elif connectivity == 2:
        structure = np.array([[1,1,1],
                              [1,1,1],
                              [1,1,1]])

    xs, ys = [], []
    for task_idx in range(10000):
        inp = np.random.randint(low=0, high=2, size=(20, 1, 32, 32)).astype(np.float32)
        result = np.expand_dims(np.array([skimage.ndimage.label(vec, structure)[1] for vec in inp]), (1,2))
        xs.append(inp)
        ys.append(result)
        if task_idx % 100 == 0:
            print(f"Finished generating task {task_idx}")
    xs, ys = np.stack(xs), np.stack(ys)
    ws = np.zeros_like(ys)
    np.savez(out_path, x=xs, y=ys, w=ws)


TYPE_2_PATH = {
    "rank1": "./data/rank1.npz",
    "rank2": "./data/rank2.npz",
    "rank5": "./data/rank5.npz",
    "2d_rot8": "./data/2d_rot8.npz",
    "2d_rot8_flip": "./data/2d_rot8_flip.npz",
    "1d_cc": "./data/1d_cc.npz",
    "2d_cc": "./data/2d_cc.npz",
    "1d_cc_contrastive": "./data/1d_cc_contrastive.npz",
    "2d_cc_contrastive": "./data/2d_cc_contrastive.npz",
    "1d_stack": "./data/1d_stack.npz",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="rank1")
    parser.add_argument("--length", type=int, default=70)
    args = parser.parse_args()
    out_path = f"./data/{args.problem}_{args.length}.npz"
    if os.path.exists(out_path):
        raise ValueError(f"File exists at {out_path}.")
    if args.problem == "rank1":
        generate_1d(out_path)
    elif args.problem == "rank2":
        generate_1d_low_rank(out_path, rank=2)
    elif args.problem == "rank5":
        generate_1d_low_rank(out_path, rank=5)
    elif args.problem == "2d_rot8":
        generate_2d_rot8(out_path)
    elif args.problem == "2d_rot8_flip":
        generate_2d_rot8_flip(out_path)
    elif args.problem == "1d_cc":
        generate_1d_cc(out_path)
    elif args.problem == "2d_cc":
        generate_2d_cc(out_path)
    elif args.problem == "1d_cc_contrastive":
        generate_1d_cc_contrastive(out_path, args.length)
    elif args.problem == "2d_cc_contrastive":
        generate_2d_cc_contrastive(out_path)
    elif args.problem == "1d_stack":
        generate_1d_stack(out_path)
    else:
        raise ValueError(f"Unrecognized problem {args.problem}")


if __name__ == "__main__":
    main()

