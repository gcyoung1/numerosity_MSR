"""Main training script for synthetic problems."""

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

import layers
from synthetic_loader import SyntheticLoader
from inner_optimizers import InnerOptBuilder

OUTPUT_PATH = "./outputs/synthetic_outputs"

def triplet_loss(x, pos, neg, dist='l2'):
    if dist == 'l2':
        return torch.sum(torch.linalg.norm(x - pos)) - torch.sum(torch.linalg.norm(x - neg))
    elif dist == 'cos':
        return torch.sum(-F.cosine_similarity(x, pos) + F.cosine_similarity(x,neg))

def train(step_idx, data, net, inner_opt_builder, meta_opt, n_inner_iter, contrastive, zeta, dist):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    meta_opt.zero_grad()
    for i in range(task_num):
        with higher.innerloop_ctx(
            net,
            inner_opt,
            copy_initial_weights=False,
            override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                if contrastive:
                    spt_pos = fnet(y_spt[i,:,0,:])
                    spt_neg = fnet(y_spt[i,:,1,:])
                    spt_loss = triplet_loss(spt_pred, spt_pos, spt_neg, dist)
                else:
                    spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])
            if contrastive:
                qry_pos = fnet(y_qry[i,:,0,:])
                qry_neg = fnet(y_qry[i,:,1,:])
                qry_loss = triplet_loss(qry_pred, qry_pos, qry_neg, dist)
            else:
                qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_loss +=  zeta * net[0].warp_l1()
            qry_losses.append(qry_loss.detach().cpu().numpy())
            qry_loss.backward()
    metrics = {"train_loss": np.mean(qry_losses)}
    wandb.log(metrics, step=step_idx)
    meta_opt.step()


def test(step_idx, data, net, inner_opt_builder, n_inner_iter, contrastive, zeta, dist):
    """Main meta-training step."""
    x_spt, y_spt, x_qry, y_qry = data
    task_num = x_spt.size()[0]
    querysz = x_qry.size(1)

    inner_opt = inner_opt_builder.inner_opt

    qry_losses = []
    for i in range(task_num):
        with higher.innerloop_ctx(
            net, inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides,
        ) as (
            fnet,
            diffopt,
        ):
            for _ in range(n_inner_iter):
                spt_pred = fnet(x_spt[i])
                if contrastive:
                    spt_pos = fnet(y_spt[i,:,0,:])
                    spt_neg = fnet(y_spt[i,:,1,:])
                    spt_loss = triplet_loss(spt_pred, spt_pos, spt_neg, dist)
                else:
                    spt_loss = F.mse_loss(spt_pred, y_spt[i])
                diffopt.step(spt_loss)
            qry_pred = fnet(x_qry[i])
            if contrastive:
                qry_pos = fnet(y_qry[i,:,0,:])
                qry_neg = fnet(y_qry[i,:,1,:])
                qry_loss = triplet_loss(qry_pred, qry_pos, qry_neg, dist)
            else:
                qry_loss = F.mse_loss(qry_pred, y_qry[i])
            qry_loss +=  zeta * net[0].warp_l1()
            qry_losses.append(qry_loss.detach().cpu().numpy())
    avg_qry_loss = np.mean(qry_losses)
    _low, high = st.t.interval(
        0.95, len(qry_losses) - 1, loc=avg_qry_loss, scale=st.sem(qry_losses)
    )
    test_metrics = {"test_loss": avg_qry_loss, "test_err": high - avg_qry_loss}
    wandb.log(test_metrics, step=step_idx)
    return avg_qry_loss


def save_sharemlp(net, figname, step):
    if not os.path.exists(figname):
        os.mkdir(figname)
    layer_idx = 0
    for layer in net[0].layers:
        if hasattr(layer, 'warp'):
            U, v = layer.warp.detach().cpu().numpy(), layer.latent_params.detach().cpu().numpy()
            W = layer.get_weight().detach().cpu().numpy()
            plotU, plotW = np.abs(U), np.abs(W)
            plt.imshow(plotW)
            plt.axis('off')
            plt.savefig(os.path.join(figname,f"{figname}_{step}_W_{layer_idx}.png"))
            fig, axs = plt.subplots(1, W.shape[1])
            for i in range(W.shape[1]):
                group = plotU[(i)*W.shape[0]:(i+1)*W.shape[0], :]
                axs[i].imshow(group)
                axs[i].axis('off')
            fig.savefig(os.path.join(figname, f"{figname}_{step}_U_{layer_idx}.png"))
            fig.clear()

            layer_idx += 1

    plt.close('all')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_inner_lr", type=float, default=0.1)
    parser.add_argument("--outer_lr", type=float, default=0.001)
    parser.add_argument("--k_spt", type=int, default=1)
    parser.add_argument("--k_qry", type=int, default=19)
    parser.add_argument("--lr_mode", type=str, default="per_layer")
    parser.add_argument("--num_inner_steps", type=int, default=1)
    parser.add_argument("--num_outer_steps", type=int, default=1000)
    parser.add_argument("--num_outer_steps_load", type=int, default=0)
    parser.add_argument("--inner_opt", type=str, default="maml")
    parser.add_argument("--outer_opt", type=str, default="Adam")
    parser.add_argument("--problem", type=str, default="rank1")
    parser.add_argument("--length", type=int, default=70)
    parser.add_argument("--hidden_size", type=int, default=15)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--model", type=str, default="conv")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--figname", type=str, default="")
    parser.add_argument("--zeta", type=float, default=0.0)
    parser.add_argument("--dist", type=str, default="l2")

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    wandb.init(project="weight_sharing_toy", dir=OUTPUT_PATH)
    args = parser.parse_args()
    wandb.config.update(args)
    cfg = wandb.config
    device = torch.device(cfg.device)
    db = SyntheticLoader(device, problem=cfg.problem, length=cfg.length, k_spt=cfg.k_spt, k_qry=cfg.k_qry)

    if cfg.problem in ["2d_rot8_flip", "2d_rot8", "2d_cc", "2d_cc_contrastive"]:
        c_o = 24 if cfg.problem == "2d_rot8" else 48
        if cfg.model == "share_conv":
            net = nn.Sequential(layers.ShareConv2d(1, c_o, 3, bias=False)).to(device)
        elif cfg.model == "conv":
            net = nn.Sequential(nn.Conv2d(1, c_o, 3, bias=False)).to(device)
        else:
            raise ValueError(f"Invalid model {cfg.model}")
    elif cfg.problem in ["rank1", "rank2", "rank5", "1d_cc", "1d_cc_contrastive", "1d_stack"]:
        if cfg.problem == "1d_cc":
            output_size = 1
        elif cfg.problem == "1d_cc_contrastive":
            output_size = cfg.length
        elif cfg.problem == "1d_stack":
            output_size = cfg.length-4
        else:
            output_size = cfg.length-2
        
        if cfg.model == "lc":
            net = nn.Sequential(layers.LocallyConnected1d(1, 1, output_size, kernel_size=3, bias=False)).to(
                device
            )
        elif cfg.model == "fc":
            net = nn.Sequential(nn.Linear(cfg.length, output_size, bias=False)).to(device)
        elif cfg.model == "conv":
            assert cfg.problem != "1d_cc"
            net = nn.Sequential(nn.Conv1d(1, 1, kernel_size=3, bias=False)).to(device)
        elif cfg.model == "share_fc":
            latent = {"rank1": 3, "rank2": 6, "rank5": 30, "1d_cc": 30, "1d_cc_contrastive": cfg.length//2}[cfg.problem]
            net = nn.Sequential(layers.ShareLinearFull(cfg.length, output_size, bias=False, latent_size=latent)).to(
                device
            )
        elif cfg.model == "share_mlp":
            latent = {"rank1": 3, "rank2": 6, "rank5": 30, "1d_cc": 30, "1d_cc_contrastive": 3, "1d_stack": 5}[cfg.problem]
            net = nn.Sequential(layers.ShareMLPFull(cfg.length, cfg.hidden_size, cfg.num_hidden_layers, output_size, bias=False, latent_size=latent)).to(
                device
            )
        else:
            raise ValueError(f"Invalid model {cfg.model}")

    if args.num_outer_steps_load > 0:
        net_params = np.load(f"net_params_{cfg.num_outer_steps_load}.npz")
        U, v = net_params["U"], net_params["v"]
        net[0].state_dict()['warp'].copy_(torch.from_numpy(U))
        net[0].state_dict()['latent_params'].copy_(torch.from_numpy(v))

    inner_opt_builder = InnerOptBuilder(
        net, device, cfg.inner_opt, cfg.init_inner_lr, "learned", cfg.lr_mode
    )
    if cfg.outer_opt == "SGD":
        meta_opt = optim.SGD(inner_opt_builder.metaparams.values(), lr=cfg.outer_lr)
    else:
        meta_opt = optim.Adam(inner_opt_builder.metaparams.values(), lr=cfg.outer_lr)

    start_time = time.time()
    val_losses = []
    for step_idx in range(cfg.num_outer_steps):
        data, _filters = db.next(32, "train")
        contrastive = "contrastive" in cfg.problem
        train(step_idx, data, net, inner_opt_builder, meta_opt, cfg.num_inner_steps, contrastive, args.zeta, args.dist)
        if step_idx == 0 or (step_idx + 1) % 100 == 0:
            test_data, _filters  = db.next(300, "test")
            val_loss = test(
                step_idx,
                test_data,
                net,
                inner_opt_builder,
                cfg.num_inner_steps,
                contrastive,
                args.zeta,
                args.dist
            )
            val_losses.append(val_loss)
            if step_idx > 0:
                steps_p_sec = (step_idx + 1) / (time.time() - start_time)
                wandb.log({"steps_per_sec": steps_p_sec}, step=step_idx)
                print(f"Step: {step_idx}. Steps/sec: {steps_p_sec:.2f}")
        if step_idx % 1000 == 0:
            torch.save(net.state_dict(), os.path.join(args.figname, 'final_weights.pth'))
            if cfg.model == "share_mlp":
                save_sharemlp(net, args.figname, step_idx)

            elif cfg.model == "share_fc":
                if not os.path.exists(args.figname):
                    os.mkdir(args.figname)

                U, v, W = net[0].warp.detach().numpy(), net[0].latent_params.detach().numpy(), net[0].get_weight().detach().numpy()
                total_steps = cfg.num_outer_steps + cfg.num_outer_steps_load
                plotU, plotW = np.abs(U), np.abs(W)
                plt.imshow(W)
                plt.axis('off')
                plt.savefig(f"{args.figname}_W_{step_idx}.png")

                fig, axs = plt.subplots(1, W.shape[1])
                for i in range(W.shape[1]):
                    group = plotU[(i)*W.shape[0]:(i+1)*W.shape[0], :]
                    axs[i].imshow(group)
                    axs[i].axis('off')
                fig.savefig(os.path.join(args.figname, f"{args.figname}_U_{step_idx}.png"))
                plt.close('all')

    print(val_losses)

if __name__ == "__main__":
    main()
