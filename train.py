import argparse
import time
import typing

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import os
import  torchvision.transforms as transforms
from    PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import higher
import logging

from omniglot_loaders import OmniglotNShot, Omniglot


from model import Net, ICM
from helper import plot, compute_icm_loss

def main():

    logging.basicConfig(level=logging.INFO)


    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument(
        '--k_spt', type=int, help='k shot for support set', default=5)
    argparser.add_argument(
        '--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument(
        '--task_num',
        type=int,
        help='meta batch size, namely task num',
        default=32)
    argparser.add_argument('--seed', type=int, help='random seed', default=1)
    argparser.add_argument("--device", type=str, help="Device to use (cpu, cuda, mps)", default="mps")
    argparser.add_argument("--inners_train", type=int, help="Inner loop updates.", default=5)
    argparser.add_argument("--inners_test", type=int, help="Inner loop updates.", default=5)
    argparser.add_argument("--model_name", type=str, help="Model name (maml, icm-maml, naive)", default="maml")
    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Set up the Omniglot loader.
    
    if args.device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device("cpu")
    

    logging.info(f"Using device {device}")
    
    db = OmniglotNShot(
        '/tmp/omniglot-data',
        batchsz=args.task_num,
        n_way=args.n_way,
        k_shot=args.k_spt,
        k_query=args.k_qry,
        imgsz=28,
        device=device,
    )

    if args.model_name == 'naive':
        # load the whole omniglot dataset
        data = Omniglot(
        '/tmp/omniglot-data',
        download=False,
        transform=transforms.Compose(
                    [lambda x: Image.open(x).convert('L'),
                     lambda x: x.resize((28, 28)),
                     lambda x: np.reshape(x, (28, 28, 1)),
                     lambda x: np.transpose(x, [2, 0, 1]),
                     lambda x: x/255.])
        )
        # split the dataset into train and test, but keep the first 1200 classes for training
        data = torch.utils.data.random_split(data, [1200, len(data) - 1200])[0]
        data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)


    # Create a vanilla PyTorch neural network that will be
    net = Net(args, device)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(net.parameters(), lr=1e-4)

    # Create the ICM model
    icm_model = ICM(input_dim=75205, hidden_dim=1024, action_dim=1024, fwd_hidden=1024, device=device)
    icm_opt = optim.SGD(icm_model.parameters(), lr=1e-3)


    log = []
    for epoch in range(100):

        # Train the model
        if args.model_name == 'maml':
            train(db, net, device, meta_opt, epoch, log, inner_iters=args.inners_train)
            test(db, net, device, epoch, log, inner_iters=args.inners_test)
        elif args.model_name == 'icm-maml':
            train_curiosity(db, net, device, meta_opt, epoch, log, icm_model, icm_opt, inner_iters=args.inners_train)
            test(db, net, device, epoch, log, inner_iters=args.inners_test)
        elif args.model_name == 'naive':
            # Naive ==> just pretrain the model on the support set and test on the query set.
            train_naive(data_loader, net, device, meta_opt, epoch, log)
            test(db, net, device, epoch, log, inner_iters=args.inners_test)
        

        # Plot the results.
        plot(log, setting_name=f"{args.n_way}-way-{args.k_spt}-shot", model_name=f"{args.model_name}")

        # Save the log.
        os.makedirs('logs', exist_ok=True)
        with open(f'logs/{args.model_name}-{args.n_way}-way-{args.k_spt}-log.pkl', 'wb') as f:
            pd.to_pickle(log, f)

def train_curiosity(db, net, device, meta_opt, epoch, log, icm_model, icm_opt, inner_iters=5):
    net.train()
    n_train_iter = db.x_train.shape[0] // db.batchsz

    qry_losses = []
    qry_accs = []

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next(mode='train')

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        n_inner_iter = inner_iters
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        icm_losses = []
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, copy_initial_weights=False) as (fmodel, diff_optim):
                # Perform a inner loop iteration.
                for _ in range(n_inner_iter):
                    # forward pass
                    spt_logits = fmodel(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    
                    # compute state transitions
                    s_t = torch.cat([p.flatten() for p in fmodel.parameters()])
                    grads = torch.autograd.grad(spt_loss, fmodel.parameters(), create_graph=True)
                    a_t = torch.cat([g.flatten() for g in grads])  # Treat gradients as actions
                    
                    # simulate weight update without applying it yet (to compute s_t1)
                    s_t1 = s_t - torch.cat([g.flatten() * inner_opt.param_groups[0]['lr'] for g in grads])

                    # compute total loss
                    icm_loss, forward_loss = compute_icm_loss(s_t, s_t1, a_t, icm_model)
                    icm_losses.append(icm_loss.detach())
                    
                    # gradient step on the curiosity model
                    icm_opt.zero_grad()
                    icm_loss.backward()
                    icm_opt.step()

                    # compute the total loss via the curiosity loss
                    _lambda = 4
                    forward_error = torch.log(forward_loss.detach() + 1.0)
                    total_loss = spt_loss + _lambda * forward_error

                    # perform a single weight update with the total loss
                    diff_optim.step(total_loss)
                
                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fmodel(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                qry_loss.backward()


        # Perform a meta-optimization step to update the model
        meta_opt.step()
        icm_losses = sum(icm_losses) / task_num
        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(
                f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Curiosity Loss: {icm_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
            )

        log.append({
            'epoch': i,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })


def train(db, net, device, meta_opt, epoch, log, inner_iters=5):
    net.train()
    n_train_iter = db.x_train.shape[0] // db.batchsz

    for batch_idx in range(n_train_iter):
        start_time = time.time()
        # Sample a batch of support and query images and labels.
        x_spt, y_spt, x_qry, y_qry = db.next()

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?

        # Initialize the inner optimizer to adapt the parameters to
        # the support set.
        n_inner_iter = inner_iters
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        qry_losses = []
        qry_accs = []
        meta_opt.zero_grad()
        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                qry_logits = fnet(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i])
                qry_losses.append(qry_loss.detach())
                qry_acc = (qry_logits.argmax(dim=1) == y_qry[i]).sum().item() / querysz
                qry_accs.append(qry_acc)

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                # accumulates gradients from the most recent forward step
                qry_loss.backward()

        # Perform a meta-optimization step to update the model
        meta_opt.step()
        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100. * sum(qry_accs) / task_num
        i = epoch + float(batch_idx) / n_train_iter
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(
                f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
            )

        log.append({
            'epoch': i,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'train',
            'time': time.time(),
        })

def train_naive(dataloader, net, device, meta_opt, epoch, log):
    net.train()
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device).float(), y.to(device).long()
        start_time = time.time()
        loss = []
        acc = []

        meta_opt.zero_grad()
        logits = net(x)
        # Ensure target labels are within the range of the number of classes
        y = torch.clamp(y, 0, logits.size(1) - 1)
        i_loss = F.cross_entropy(logits, y)
        i_acc = (logits.argmax(dim=1) == y).sum().item() / x.size(0)
        acc.append(i_acc)
        i_loss.backward()
        loss.append(i_loss.item())

        # Perform a step to update the model
        meta_opt.step()
        loss = sum(loss) / len(loss)
        acc = 100. * sum(acc) / len(acc)
        i = epoch + float(batch_idx) / dataloader.__len__()
        iter_time = time.time() - start_time
        if batch_idx % 4 == 0:
            print(
                f'[Epoch {i:.2f}] Train Loss: {loss:.2f} | Acc: {acc:.2f} | Time: {iter_time:.2f}'
            )

        log.append({
            'epoch': i,
            'loss': loss,
            'acc': acc,
            'mode': 'train',
            'time': time.time(),
        })



def test(db, net, device, epoch, log, inner_iters=5):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.
    net.train()
    n_test_iter = db.x_test.shape[0] // db.batchsz

    qry_losses = []
    qry_accs = []

    for batch_idx in range(n_test_iter):
        x_spt, y_spt, x_qry, y_qry = db.next('test')


        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # TODO: Maybe pull this out into a separate module so it
        # doesn't have to be duplicated between `train` and `test`?
        n_inner_iter = inner_iters
        inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        for i in range(task_num):
            with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                for _ in range(n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # The query loss and acc induced by these parameters.
                qry_logits = fnet(x_qry[i]).detach()
                qry_loss = F.cross_entropy(qry_logits, y_qry[i], reduction='none')
                qry_losses.append(qry_loss.detach())
                qry_accs.append((qry_logits.argmax(dim=1) == y_qry[i]).detach())

    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
    print(
        f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
    )
    log.append({
        'epoch': epoch + 1,
        'loss': qry_losses,
        'acc': qry_accs,
        'mode': 'test',
        'time': time.time(),
    })


if __name__ == '__main__':
    main()