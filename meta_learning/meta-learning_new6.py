"""
Meta-learning Omniglot and mini-imagenet experiments with iMAML-GD (see [1] for more details).

The code is quite simple and easy to read thanks to the following two libraries which need both to be installed.
- higher: https://github.com/facebookresearch/higher (used to get stateless version of torch nn.Module-s)
- torchmeta: https://github.com/tristandeleu/pytorch-meta (used for meta-dataset loading and minibatching)


[1] Rajeswaran, A., Finn, C., Kakade, S. M., & Levine, S. (2019).
    Meta-learning with implicit gradients. In Advances in Neural Information Processing Systems (pp. 113-124).
    https://arxiv.org/abs/1909.04630
"""
import os
import h5py

import math
import argparse
import time
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchmeta.datasets.helpers import omniglot, miniimagenet,doublemnist
from torchmeta.utils.data import BatchMetaDataLoader

import sys
sys.path.append("..")

import hypergrad as hg
import higher

#import hypergrad as hg
import csv
import psutil as psutil
from torch.autograd import grad as torch_grad
from hypergrad.hypergradients import list_tensor_norm, get_outer_gradients, list_tensor_matmul, update_tensor_grads, \
    grad_unused_zero

torch.cuda.set_device(0)


filename = "./MEHA.h5"
bome_iter_list = []
bome_acc_list = []
bome_time_list = []

class Task:
    """
    Handles the train and valdation loss for a single task
    """

    def __init__(self, reg_param, meta_model_x, meta_model_y, data, batch_size=None, alpha=0.5, BDA=False):
        device = next(meta_model_y.parameters()).device

        # stateless version of meta_model
        self.fmodel = higher.monkeypatch(meta_model_y, device=device, copy_initial_weights=True)
        self.meta_model_x = meta_model_x
        self.n_params = len(list(meta_model_y.parameters()))
        self.train_input, self.train_target, self.test_input, self.test_target = data
        self.reg_param = reg_param
        self.batch_size = 1 if not batch_size else batch_size
        self.val_loss, self.val_acc = None, None
        self.alpha = alpha
        self.BDA = BDA

    def bias_reg_f(self, bias, params):
        # l2 biased regularization
        return sum([((b - p) ** 2).sum() for b, p in zip(bias, params)])

    def train_loss_f(self, params, hparams):
        # biased regularized cross-entropy loss where the bias are the meta-parameters in hparams
        out = self.fmodel(self.meta_model_x(self.train_input), params=params)
        return F.cross_entropy(out, self.train_target)  # + 0.5 * self.reg_param * self.bias_reg_f(hparams, params)

    def train_loss_f_BDA(self, params, hparams):
        # biased regularized cross-entropy loss where the bias are the meta-parameters in hparams
        out = self.fmodel(self.meta_model_x(self.train_input), params=params)
        f_loss = F.cross_entropy(out, self.train_target)
        out = self.fmodel(self.meta_model_x(self.test_input), params=params)
        val_loss = F.cross_entropy(out, self.test_target) / self.batch_size
        return (
                           1 - self.alpha) * f_loss + self.alpha * val_loss  # + 0.5 * self.reg_param * self.bias_reg_f(hparams, params)

    def val_loss_f(self, params, hparams):
        # cross-entropy loss (uses only the task-specific weights in params
        out = self.fmodel(self.meta_model_x(self.test_input), params=params)
        val_loss = F.cross_entropy(out, self.test_target) / self.batch_size
        self.val_loss = val_loss.item()  # avoid memory leaks

        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        self.val_acc = pred.eq(self.test_target.view_as(pred)).sum().item() / len(self.test_target)

        return val_loss

    def low_loss_FO(self,theta, params, hparams, ck, gamma):
        return (ck) * self.val_loss_f(params, hparams) + self.val_loss_f(params, hparams) + 0.5 * gamma * self.bias_reg_f(params,theta)

    def low_loss_theta(self,theta, params, hparams, gamma):

        return self.val_loss_f(theta, hparams) + 0.5 * gamma * self.bias_reg_f(params,theta)


    def upper_loss_FO(self,theta, params, hparams, ck):
        return (ck) * self.val_loss_f(params, hparams) + self.train_loss_f(params, hparams) - self.train_loss_f(theta, hparams)


    def low_loss_gamma(self,theta, params, hparams, gamma):
        reg = 0
        for param1, param2 in zip(params[-6:-1], theta[-6:-1]):
            diff = param1 - param2
            # result_params.append(diff)
            reg += torch.norm(diff, p=2) ** 2
        return self.train_loss_f(theta, hparams)+ 0.5 * gamma * reg


class MyDataloder:
    def __init__(self,train_path,val_path,test_path,rand_way=True,shuffle=True):
        self.train_path=train_path
        self.val_path=val_path
        self.test_path=test_path
        self.rand_way=rand_way
        self.shuffle=shuffle

def copy_parameter_from_list(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y

def main():
    parser = argparse.ArgumentParser(description='Data HyperCleaner')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ways', type=int, default=10)
    parser.add_argument('--shots', type=int, default=1)
    parser.add_argument('--T0', type=int, default=10)
    parser.add_argument('--T_test', type=int, default=10)
    parser.add_argument('--K0', type=int, default=5)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--log_interval',type=int,default=100)
    #parser.add_argument('--K0', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='omniglot', metavar='N', help='omniglot or miniimagenet')
    parser.add_argument('--hg-mode', type=str, default='newmethod4', metavar='N',
                        help='hypergradient approximation: CG or fixed_point')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--BDA', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--eta', type=float, default=10.)
    parser.add_argument('--tau', type=float, default=1.e-4)
    parser.add_argument('--thelr', action='store_true', default=False)
    parser.add_argument('--eta0', type=float, default=10.)  # 0.05
    parser.add_argument('--mu0', type=float, default=0.9)
    parser.add_argument('--p0', type=float, default=100)
    parser.add_argument('--tol', type=float, default=1.e-10)
    parser.add_argument('--x_lr', type=float, default=0.01)
    parser.add_argument('--y_lr', type=float, default=0.1)
    parser.add_argument('--Notes', type=str, default='s1', help='s1')
    args = parser.parse_args()

    log_interval = args.log_interval
    eval_interval = args.eval_interval
    inner_log_interval = None
    inner_log_interval_test = None
    ways = args.ways
    batch_size = 16
    n_tasks_test = 1000  # usually 1000 tasks are used for testingS
    if args.dataset == 'omniglot':
        reg_param = 2  # reg_param = 2
        T, K = args.T0, args.K0  # T, K = 16, 5
    elif args.dataset == 'miniimagenet':
        reg_param = 0.5  # reg_param = 0.5
        T, K = 10, 5  # T, K = 10, 5
    elif args.dataset == 'doublemnist':
        reg_param = 2  # reg_param = 2
        T, K = 10, 5  # T, K = 16, 5
    else:
        raise NotImplementedError(args.dataset, " not implemented!")

    T_test = copy.deepcopy(args.T_test)
    inner_lr = args.y_lr

    if args.hg_mode.find('Darts') != -1:
        print('One Stage')
        T= 1
    T = 1

    loc = locals()
    del loc['parser']
    del loc['args']

    print(args, '\n', loc, '\n')

    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # log_path = "metalearning_{}_{}_BDA{}{:.2f}_ways{}_shots{}_Notes{}_{}.csv".format(args.hg_mode, args.dataset, args.BDA, args.alpha,args.ways,args.shots,args.Notes,
    #                                                                    time.strftime("%Y_%m_%d_%H_%M_%S"))
    # with open(log_path, 'a', encoding='utf-8') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow([args])
    #     csv_writer.writerow(['', 'acc','total_hyper_time', 'total_time', 'Val loss'])
    # the following are for reproducibility on GPU, see https://pytorch.org/docs/master/notes/randomness.html
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'omniglot':
        dataset = omniglot(r"./dataset", ways=args.ways, shots=args.shots, test_shots=15, meta_train=True, download=True)
        test_dataset = omniglot(r"./dataset", ways=args.ways, shots=args.shots, test_shots=15, meta_test=True, download=True)

        meta_model_x, meta_model_y = get_cnn_omniglot(64, args.ways)
        meta_model_x = meta_model_x.to(device)
        meta_model_y = meta_model_y.to(device)
    elif args.dataset=='doublemnist':
        dataset = doublemnist(r"./dataset", ways=args.ways, shots=args.shots,
                           test_shots=15, meta_train=True, download=True)
        test_dataset = doublemnist(r"./dataset", ways=args.ways, shots=args.shots,
                                test_shots=15, meta_test=True, download=True)

        meta_model_x, meta_model_y = get_cnn_dmnist(32, args.ways)
        meta_model_x = meta_model_x.to(device)
        meta_model_y = meta_model_y.to(device)
    elif args.dataset == 'miniimagenet':
        dataset = miniimagenet(r"./dataset", ways=args.ways, shots=args.shots, test_shots=15,
                               meta_train=True, download=True)
        test_dataset = miniimagenet(r"./dataset", ways=args.ways, shots=args.shots, test_shots=15,
                                    meta_test=True, download=True)

        meta_model_x, meta_model_y = get_cnn_miniimagenet(32, args.ways)
        meta_model_x = meta_model_x.to(device)
        meta_model_y = meta_model_y.to(device)
    else:
        raise NotImplementedError("DATASET NOT IMPLEMENTED! only omniglot and miniimagenet ")

    dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, **kwargs)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=batch_size, **kwargs)

    outer_opt = torch.optim.Adam(params=meta_model_x.parameters(),lr=args.x_lr)
    # outer_opt = torch.optim.SGD(lr=0.1, params=meta_model.parameters())
    inner_opt_class = hg.GradientDescent2


    inner_opt_kwargs = {'step_size': inner_lr}

    def get_inner_opt(train_loss):
        return inner_opt_class(train_loss, **inner_opt_kwargs)
    total_time=0
    # vs=-1
    v=-1
    ck = -1
    for k, batch in enumerate(dataloader):
        start_time = time.time()
        meta_model_x.train()
        initialize(meta_model_y)

        tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
        tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)
        if args.thelr:
            args.alpha = args.mu0 * 1 / (k + 1) ** (1 / args.p0)

            # args.x_lr=args.alpha0*args.alpha**11

            # eta=args.eta0*args.alpha**4
            eta = (k + 1) ** (-0.5 * args.tau) * args.alpha ** 2 * args.y_lr
            args.x_lr = (k + 1) ** (-1.5 * args.tau) * args.alpha ** 7 * args.y_lr
            for params in outer_opt.param_groups:
                params['lr'] = args.x_lr

        outer_opt.zero_grad()

        val_loss, val_acc = 0, 0
        forward_time, backward_time = 0, 0
        y_time,v_time,x_time,prepare_time=0,0,0,0
        for t_idx, (tr_x, tr_y, tst_x, tst_y) in enumerate(zip(tr_xs, tr_ys, tst_xs, tst_ys)):
            start_time_task = time.time()
            t0 = time.time()
            gamma_1 = 0.001
            if ck == -1 or t_idx ==0:
                ck = c0 = 5
            else:
                ck = np.power(t_idx + 1, 0.1) * c0
            # single task set up
            task = Task(reg_param, meta_model_x, meta_model_y, (tr_x, tr_y, tst_x, tst_y), batch_size=tr_xs.shape[0],
                        alpha=args.alpha, BDA=args.BDA)
            inner_opt = get_inner_opt(task.train_loss_f if not args.BDA else task.train_loss_f_BDA)
            new_time = time.time()
            # single task inner loop
            params = [p.detach().clone().requires_grad_(True) for p in meta_model_y.parameters()]
            # print(T if args.hg_mode != 'Darts_W' else 1)

            vs = [torch.zeros_like(w).requires_grad_(True) for w in params] if v == -1 else v


            y_time_task = time.time() - new_time

            forward_time_task = time.time() - start_time_task

            # single task hypergradient computation
            if args.hg_mode == 'CG':
                # This is the approximation used in the paper CG stands for conjugate gradient
                cg_fp_map = hg.GradientDescent(loss_f=task.train_loss_f, step_size=1.)
                hg.CG(last_param, list(meta_model_x.parameters()), K=K, fp_map=cg_fp_map, outer_loss=task.val_loss_f)
            elif args.hg_mode == 'fixed_point':
                hg.fixed_point(last_param, list(meta_model_x.parameters()), K=K, fp_map=inner_opt,
                               outer_loss=task.val_loss_f,tol=args.tol)
            elif args.hg_mode == 'RHG':
                hg.reverse(last_param, list(meta_model_x.parameters()),  [inner_opt] * T,
                               outer_loss=task.val_loss_f)
            elif args.hg_mode == 'Darts_W_RHG':
                grads,vs=hg.Darts_W_RHG(last_param, list(meta_model_x.parameters()), K=K, fp_map=inner_opt,
                           outer_loss=task.val_loss_f,v0=v,ita=eta)
                v=vs
            elif args.hg_mode =='Darts_W_RHG_mp':
                #params = [w.detach().requires_grad_(True) for w in list(meta_model_y.parameters())]  # y
                params = [w.detach().requires_grad_(True) for w in last_param]  # y
                o_loss = task.val_loss_f(params, list(meta_model_x.parameters()))  # F
                grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params,
                                                                       list(meta_model_x.parameters()))  # dy F,dx F
                w_mapped = inner_opt(params, list(meta_model_x.parameters()), only_grad=True)  # dy f
                prepare_time_task = time.time() - t0 - y_time_task
                hyper_time = time.time()
                vs = [torch.zeros_like(w) for w in params] if v == -1 else v
                vsp = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True,
                                 allow_unused=True)  # dy (dy f) v=d2y f v
                vs = [v0 - eta * (v if v is not None else 0) + eta * (gow if gow is not None else 0) for v0, v, gow in
                      zip(vs, vsp, grad_outer_w)]  # (I-ita*d2yf)v+ita*dy F)
                v_time_task = time.time() - hyper_time
                new_time = time.time()
                grads = torch_grad(w_mapped, list(meta_model_x.parameters()),
                                   grad_outputs=[torch.zeros_like(w) for w in params] if v == -1 else v,
                                   allow_unused=True)  # dx (dy f) v
                grads = [-g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

                update_tensor_grads(list(meta_model_x.parameters()), grads)
                x_time_task = time.time() - new_time
                v = vs
                #gts = grads
            elif args.hg_mode=='Darts_W_CG_mp':
                params = [w.detach().requires_grad_(True) for w in list(meta_model_y.parameters())]  # y
                o_loss = task.val_loss_f(params, list(meta_model_x.parameters()))  # F
                grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params,
                                                                       list(meta_model_x.parameters()))  # dy F,dx F
                w_mapped = inner_opt(params, list(meta_model_x.parameters()), only_grad=True)  # dy f
                prepare_time_task = time.time() - t0 - y_time_task
                hyper_time = time.time()
                vs = [torch.zeros_like(w) for w in params] if v == -1 else v

                vsp = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True,
                                 allow_unused=True)  # dy (dy f) v=d2y f v
                tem = [v - gow for v, gow in zip(vsp, grad_outer_w)]

                ita_u = list_tensor_norm(tem) ** 2
                grad_tem = torch_grad(w_mapped, params, grad_outputs=tem, retain_graph=True,
                                      allow_unused=True)  # dy (dy f) v=d2y f v

                ita_l = list_tensor_matmul(tem, grad_tem, trans=1)
                # print(ita_u,ita_l)
                ita = ita_u / (ita_l + 1e-12)
                vs = [v0 - ita * v + ita * gow for v0, v, gow in zip(vs, vsp, grad_outer_w)]  # (I-ita*d2yf)v+ita*dy F)

                v_time_task = time.time() - hyper_time
                new_time = time.time()
                grads = torch_grad(w_mapped, list(meta_model_x.parameters()),
                                   grad_outputs=[torch.zeros_like(w) for w in params] if v == -1 else v,
                                   allow_unused=True)  # dx (dy f) v

                grads = [-g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

                # if set_grad:
                update_tensor_grads(list(meta_model_x.parameters()), grads)

                x_time_task = time.time() - new_time
                v = vs
                eta = ita

            elif args.hg_mode == 'newmethod4':
                # hg.Darts_W_RHG(last_param[-1], list(x.parameters()), K=20, fp_map=inner_opt, outer_loss=val_loss, ita=eta)
                eta = 0.001

                t0 = time.time()

                #### Update \theta We assuptation v
                params = [w.detach().requires_grad_(True) for w in list(meta_model_y.parameters())]  # y

                theta_loss = task.low_loss_theta(vs,params,meta_model_x.parameters(),gamma_1)
                grad_theta_parmaters = grad_unused_zero(theta_loss, vs)
                t1 = time.time()
                t2 = time.time()
                vs = [v0 - eta * (gt) for v0, gt in
                      zip(vs, grad_theta_parmaters)]  # upate \theta

                t2 = time.time()


                t3 = time.time()

                ## 直接对y求导
                # lower_loss = task.low_loss_FO(vs, params, meta_model_x.parameters(), ck, gamma_1)
                lower_loss = task.low_loss_FO(vs, list(params), meta_model_x.parameters(), ck, gamma_1)
                # fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                last_param = inner_loop2(lower_loss, meta_model_x.parameters(), params, inner_opt, T,
                                         log_interval=inner_log_interval)

                t4 = time.time()
                if not args.hg_mode == 'RHG':
                    last_param = last_param[-1]
                # grad_y_parmaters = grad_unused_zero(lower_loss, params)

                # update_tensor_grads(list(y.parameters()), grad_y_parmaters)

                # y_opt.step()
                # copy_parameter_from_list(meta_model_y, last_param)

                # params = [w.detach().requires_grad_(True) for w in list(last_param)]  # y

                upper_loss = task.upper_loss_FO(vs, last_param, meta_model_x.parameters(), ck)
                grad_x_parmaters = grad_unused_zero(upper_loss, list(meta_model_x.parameters()))

                update_tensor_grads(list(meta_model_x.parameters()), grad_x_parmaters)

                x_time = time.time() - new_time
                v = vs

            elif args.hg_mode == 'Darts_W_CG':
                grads,vs,eta=hg.Darts_W_CG(last_param, list(meta_model_x.parameters()), K=K, fp_map=inner_opt,
                               outer_loss=task.val_loss_f)

            backward_time_task = time.time() - start_time_task - forward_time_task

            val_loss += task.val_loss
            val_acc += task.val_acc / task.batch_size

            forward_time += forward_time_task
            backward_time += backward_time_task
            if 'mp' in args.hg_mode:
                y_time+=y_time_task
                v_time+=v_time_task
                x_time+=x_time_task
                prepare_time+=prepare_time_task

        time_update=time.time()
        outer_opt.step()
        x_time+=time.time()-time_update
        if 'mp' in args.hg_mode:
            time_list = [prepare_time + x_time+y_time, prepare_time + v_time+y_time]
            outer_time = max(time_list)
            step_time = outer_time
            #total_hyper_time += prepare_time + x_time + v_time
            total_time += outer_time
        else:
            step_time = time.time() - start_time
            total_time=total_time+step_time

        if k % log_interval == 0:
            print('MT k={} ({:.3f}s F: {:.3f}s, B: {:.3f}s) Val Loss: {:.2e}, Val Acc: {:.2f},Total time: {:.2f}.'
                  .format(k, step_time, forward_time, backward_time, val_loss, 100. * val_acc,total_time))

        if k % eval_interval == 0:
            test_losses, test_accs = evaluate(n_tasks_test, test_dataloader, meta_model_x, meta_model_y, T_test,
                                              get_inner_opt,
                                              reg_param, log_interval=inner_log_interval_test,args=args)

            print("Test loss {:.2e} +- {:.2e}: Test acc: {:.2f} +- {:.2e} (mean +- std over {} tasks).,Total time: {:.2f}."
                  .format(test_losses.mean(), test_losses.std(), 100. * test_accs.mean(),
                          100. * test_accs.std(), len(test_losses),total_time))
            bome_iter_list.append(k)
            bome_acc_list.append(100. * test_accs.mean())
            bome_time_list.append(total_time)
            # with open(log_path, 'a', encoding='utf-8', newline='') as f:
            #     csv_writer = csv.writer(f)
            #     csv_writer.writerow([k, 100. * test_accs.mean(), backward_time, step_time,test_losses.mean(),total_time])
            if 100. * test_accs.mean()>89.5:
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('iteration', data=bome_iter_list)
                    f.create_dataset('accuracy', data=bome_acc_list)
                    f.create_dataset('convergence_time', data=bome_time_list)

                print(f"Data saved to {filename}")

                break

def inner_loop(hparams, params, optim, n_steps, log_interval, create_graph=False):
    params_history = [optim.get_opt_params(params)]

    for t in range(n_steps):
        params_history.append(optim(params_history[-1], hparams, create_graph=create_graph))

        if log_interval and (t % log_interval == 0 or t == n_steps - 1):
            print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

    return params_history


def inner_loop2(loss, hparams, params, optim, n_steps, log_interval, create_graph=False):
    params_history = [optim.get_opt_params(params)]

    for t in range(n_steps):
        params_history.append(optim(loss, params_history[-1], hparams, create_graph=create_graph))
        if log_interval and (t % log_interval == 0 or t == n_steps - 1):
            print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))
    return params_history





def evaluate(n_tasks, dataloader, meta_model_x, meta_model_y, n_steps, get_inner_opt, reg_param, log_interval=None,args=None):
    meta_model_y.train()
    device = next(meta_model_y.parameters()).device

    val_losses, val_accs = [], []
    for k, batch in enumerate(dataloader):
        tr_xs, tr_ys = batch["train"][0].to(device), batch["train"][1].to(device)
        tst_xs, tst_ys = batch["test"][0].to(device), batch["test"][1].to(device)
        initialize(meta_model_y)

        for t_idx, (tr_x, tr_y, tst_x, tst_y) in enumerate(zip(tr_xs, tr_ys, tst_xs, tst_ys)):
            task = Task(reg_param, meta_model_x, meta_model_y, (tr_x, tr_y, tst_x, tst_y),
                        alpha=args.alpha, BDA=args.BDA)
            inner_opt = get_inner_opt(task.train_loss_f)

            params = [p.detach().clone().requires_grad_(True) for p in meta_model_y.parameters()]
            lower_loss = task.train_loss_f(params,meta_model_x.parameters())
            last_param = inner_loop2(lower_loss,meta_model_x.parameters(), params, inner_opt, n_steps, log_interval=log_interval)[
                -1]

            task.val_loss_f(last_param, meta_model_x.parameters())

            val_losses.append(task.val_loss)
            val_accs.append(task.val_acc)

            if len(val_accs) >= n_tasks:
                return np.array(val_losses), np.array(val_accs)


def get_cnn_omniglot(hidden_size, n_classes):
    def conv_layer(ic, oc, ):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=True  # When this is true is called the "transductive setting"
                           )
        )

    net2_x = nn.Sequential(
        conv_layer(1, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten())
    net2_y = nn.Linear(hidden_size, n_classes)

    # initialize(net2_x)
    # initialize(net2_y)
    # initialize(net2_z)
    return net2_x, net2_y

def get_cnn_dmnist(hidden_size, n_classes):
    def conv_layer(ic, oc, ):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=True  # When this is true is called the "transductive setting"
                           )
        )

    net2_x = nn.Sequential(
        conv_layer(3, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten())
    net2_y = nn.Linear(hidden_size * 4* 4, n_classes)

    # initialize(net2_x)
    # initialize(net2_y)
    # initialize(net2_z)
    return net2_x, net2_y



def get_cnn_miniimagenet(hidden_size, n_classes):
    def conv_layer(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=False  # When this is true is called the "transductive setting"
                           )
        )

    net2_x = nn.Sequential(
        conv_layer(3, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten())
    net2_y = nn.Linear(hidden_size * 5 * 5, n_classes)

    initialize(net2_x)
    initialize(net2_y)

    return net2_x, net2_y


def initialize(net):
    # initialize weights properly
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # m.weight.data.normal_(0, 0.01)
            # m.bias.data = torch.ones(m.bias.data.size())
            m.weight.data.zero_()
            m.bias.data.zero_()

    return net


if __name__ == '__main__':
    main()
