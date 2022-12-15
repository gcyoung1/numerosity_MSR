"""Evaluate trained contrastive models."""

import argparse
import os
import time
import scipy.stats as st
import wandb
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import higher
import matplotlib
import matplotlib.pyplot as plt 
import scipy
import sklearn
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

import layers
from synthetic_loader import SyntheticLoader
from inner_optimizers import InnerOptBuilder

OUTPUT_PATH = "./outputs/synthetic_outputs"

def gen_all_binary_vectors(length=10):
    if length == 1:
        return [[0.],[1.]]
    less1 = gen_all_binary_vectors(length-1)
    return [[0.] + x for x in less1] + [[1.] + x for x in less1]  

def pairwise_dists(X, dist='l2'):
    if dist == 'l2':
        return pairwise_distances(X, metric='l2')
    elif dist == 'cos':
        return pairwise_distances(X, metric='cosine')

def rdm(net, ccfig, figname):
    binary_vectors = gen_all_binary_vectors(10)
    num_ccs = [scipy.ndimage.label(x, [1,1,1])[1] for x in binary_vectors]

    # Pairwise dists
    sorted_binary_vectors = [x[1] for x in sorted(zip(num_ccs, binary_vectors))]
    with torch.no_grad():
        sorted_embeddings = net(torch.tensor(sorted_binary_vectors, requires_grad=False))

    pairwise = pairwise_dists(sorted_embeddings, dist='l2')    
    plt.imshow(pairwise)
    plt.axis('off')
    plt.savefig(os.path.join(figname,f"{figname}_{ccfig}_dists.png"))
    plt.close('all')

def pca(net, ccfig, figname):
    # 2d embeddings
    binary_vectors = gen_all_binary_vectors(10)
    num_ccs = [scipy.ndimage.label(x, [1,1,1])[1] for x in binary_vectors]

    components = PCA(n_components=2).fit_transform(binary_vectors)
    num_ccs = np.array(num_ccs)
    binary_vectors = np.array(binary_vectors)
    for num_cc in np.unique(num_ccs):
        plt.scatter(components[:,0][num_ccs==num_cc],components[:,1][num_ccs==num_cc],label=num_cc)
    plt.legend()
    plt.savefig(os.path.join(figname,f"{figname}_{ccfig}_pca.png"))


def triplet_loss(x, pos, neg, dist='l2'):
    if dist == 'l2':
        return torch.sum(torch.linalg.norm(x - pos)) - torch.sum(torch.linalg.norm(x - neg))
    elif dist == 'cos':
        return torch.sum(-F.cosine_similarity(x, pos) + F.cosine_similarity(x,neg))


def test(step_idx, data, net, inner_opt_builder, n_inner_iter, contrastive, zeta, dist, figname):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    seen_ccs = set()
    for i in range(task_num):
        with higher.innerloop_ctx(
            net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            num_ccs = scipy.ndimage.label(x_spt[i][0,0,:], [1,1,1])[1]
            if num_ccs in seen_ccs: 
                continue
            seen_ccs.add(num_ccs)
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                if contrastive:
                    spt_pos = fnet(y_spt[i,:,0,:])
                    spt_neg = fnet(y_spt[i,:,1,:])
                    spt_loss = triplet_loss(spt_pred, spt_pos, spt_neg, dist)
                else:
                    spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            pca(fnet, num_ccs, figname)
            rdm(fnet, num_ccs, figname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_inner_lr", type=float, default=0.1)
    parser.add_argument("--k_spt", type=int, default=1)
    parser.add_argument("--k_qry", type=int, default=19)
    parser.add_argument("--lr_mode", type=str, default="per_layer")
    parser.add_argument("--num_inner_steps", type=int, default=1)
    parser.add_argument("--num_outer_steps", type=int, default=1000)
    parser.add_argument("--inner_opt", type=str, default="maml")
    parser.add_argument("--outer_opt", type=str, default="Adam")
    parser.add_argument("--problem", type=str, default="rank1")
    parser.add_argument("--length", type=int, default=70)
    parser.add_argument("--hidden_size", type=int, default=15)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--model", type=str, default="conv")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--figname", type=str, default="")
    parser.add_argument("--zeta", type=float, default=0.1)
    parser.add_argument("--dist", type=str, default="l2")

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.model == "share_mlp":
        net = nn.Sequential(layers.ShareMLPFull(args.length, args.hidden_size, args.num_hidden_layers, 10, bias=False, latent_size=3)).to(
            device
        )
    elif args.model == "share_fc":
        net = nn.Sequential(layers.ShareLinearFull(args.length, 10, bias=False, latent_size=5)).to(
            device
        )

    params = torch.load(os.path.join(args.figname, 'final_weights.pth'))
    net.load_state_dict(params)


    inner_opt_builder = InnerOptBuilder(
        net, device, args.inner_opt, args.init_inner_lr, "learned", args.lr_mode
    )

    db = SyntheticLoader(device, problem=args.problem, length=args.length, k_spt=args.k_spt, k_qry=args.k_qry)
    test_data, _filters  = db.next(300, "test")
    val_loss = test(
        0,
        test_data,
        net,
        inner_opt_builder,
        args.num_inner_steps,
        True,
        args.zeta,
        'l2',
        args.figname
    )


if __name__ == "__main__":
    main()
