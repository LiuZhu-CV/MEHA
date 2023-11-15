import torch
import copy
import numpy as np
import time
import csv
import argparse
import hypergrad as hg
import math
import higher
import numpy
import os
import scipy.io


import psutil as psutil
from torch.autograd import grad as torch_grad
from hypergrad.hypergradients import list_tensor_norm, get_outer_gradients, list_tensor_matmul, update_tensor_grads, \
    grad_unused_zero

parser = argparse.ArgumentParser(description='strongly_convex')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='FashionMNIST', metavar='N')
parser.add_argument('--mode', type=str, default='ours', help='ours or IFT')
parser.add_argument('--hg_mode', type=str, default='MEHA', metavar='N',
                    help='hypergradient RHG or CG or fixed_point')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--z_loop', type=int, default=10)
parser.add_argument('--y_lin_loop', type=int, default=50)
parser.add_argument('-le','--lin_error', type=float, default=-1.)
parser.add_argument('--y_loop', type=int, default=100)
parser.add_argument('--x_loop', type=int, default=300)
parser.add_argument('--z_L2_reg', type=float, default=0.01)
parser.add_argument('--y_L2_reg', type=float, default=0.0005)
parser.add_argument('--y_ln_reg', type=float, default=0.1)
parser.add_argument('--reg_decay', type=bool, default=True)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--learn_h', type=bool, default=False)
parser.add_argument('--x_lr', type=float, default=0.1)
parser.add_argument('--y_lr', type=float, default=0.00001)
parser.add_argument('--z_lr', type=float, default=0.01)
parser.add_argument('--x_lr_decay_rate', type=float, default=0.1)
parser.add_argument('--x_lr_decay_patience', type=int, default=1)
parser.add_argument('--pollute_rate', type=float, default=0.5)
parser.add_argument('--convex', type=bool, default=True)
parser.add_argument('--tSize', type=int, default=1000)
parser.add_argument('--xSize', type=int, default=1000)
parser.add_argument('--ySize', type=int, default=1000)
parser.add_argument('--log', type=int, default=1)#10
parser.add_argument('--idt', action='store_false', default=True)
parser.add_argument('--BDA', action='store_true', default=False)
parser.add_argument('--GN', action='store_true', default=False)
parser.add_argument('--linear', action='store_true', default=False)
parser.add_argument('--exp', action='store_true', default=False)
parser.add_argument('--back', action='store_true', default=False)
parser.add_argument('-ew','--element_wise', action='store_true', default=False)
parser.add_argument('--tau', type=float, default=1/1000.)
parser.add_argument('--eta_CG', type=int, default=-1)
parser.add_argument('--thelr', action='store_true', default=False)
parser.add_argument('--c', type=float, default=2.0)#0.125

parser.add_argument('--exprate', type=float, default=0.99)
parser.add_argument('--eta0', type=float, default=0.125)#0.5
parser.add_argument('-ws', '--WarmStart', action='store_false', default=True)
parser.add_argument('--eta', type=float, default=1.)
args = parser.parse_args()
if not args.WarmStart or args.hg_mode.find('Darts') != -1:
    print('One Stage')
    args.x_loop = args.x_loop * args.y_loop
    args.y_loop = 1

args.tSize=max(args.xSize,args.ySize)
if_cuda=False

print(args)

def create_tensor_with_xy(n,x0,y0):
    if n % 2 != 0:
        raise ValueError("n must be an even number")

    x = torch.ones((n // 2, 1)) *x0
    y = torch.ones((n // 2, 1)) *y0

    result_tensor = torch.cat((x, y), dim=0)

    return result_tensor


def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024
    print(f"{hint} memory used: {memory} MB ")




def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss


def loss_L1(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 1)
    return loss


def loss_L1(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 1)
    return loss


def accuary(out, target):
    pred = out.argmax(dim=1, keepdim=True)
    acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
    return acc


def Binarization(x):
    x_bi = np.zeros_like(x)
    for i in range(x.shape[0]):
        # print(x[i])
        x_bi[i] = 1 if x[i] >= 0 else 0
    return x_bi


def positive_matrix(m):
    randt = torch.rand(m) + 1
    matrix0 = torch.diag(randt)
    invmatrix0 = torch.diag(1 / randt)
    Q = torch.rand(m, m)
    Q, R = torch.qr(Q)
    matrix = torch.mm(torch.mm(Q.t(), matrix0), Q)
    invmatrix = torch.mm(torch.mm(Q.t(), invmatrix0), Q)
    return matrix, invmatrix


def semipositive_matrix(m):
    randt = torch.rand(m)
    matrix0 = torch.diag(randt)
    invmatrix0 = torch.diag(1 / randt)
    Q = torch.rand(m, m)
    Q, R = torch.qr(Q)
    matrix = torch.mm(torch.mm(Q.t(), matrix0), Q)
    invmatrix = torch.mm(torch.mm(Q.t(), invmatrix0), Q)
    return matrix, invmatrix


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


class Net_x(torch.nn.Module):
    def __init__(self, tr):
        super(Net_x, self).__init__()
        self.x = torch.nn.Parameter(torch.zeros(tr.data.shape[0]).to("cuda").requires_grad_(True))

    def forward(self, y):
        y = torch.sigmoid(self.x) * y
        y = y.mean()
        return y


def copy_parameter(y, z):
    for p, q in zip(y.parameters(), z.parameters()):
        p.data = q.clone().detach().requires_grad_()
    return y


def frnp(x):
    t = torch.from_numpy(x).cuda()
    return t


def tonp(x):
    return x.detach().cpu().numpy()


def copy_parameter(y, z):

    for p, q in zip(y.parameters(), z.parameters()):
        p.data = q.clone().detach().requires_grad_()
    return y


def copy_parameter_from_list(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y


idt = args.idt
if idt:

    A= 1
    B = 1
    D = 1
    invA = A

    xh = torch.ones([args.xSize, 1]) * 1
    ystar = create_tensor_with_xy(args.tSize,0,-1*(1.0/args.tSize)).requires_grad_(False)
    BinvATBT = 1
    inv_BinvATBT_mD = 0.5
    xstar = create_tensor_with_xy(args.tSize,1.0/args.tSize,0).requires_grad_(False)


gradlist = []
xlist = []
ylist = []
vlist = []
Flist=[]
etalist=[]
xgradlist=[]
timelist = []


x_error_list = []
y_error_list = []
time_list=[]

for x0, y0 in zip([0.2], [0.2]):
    for a, b in zip([2], [2]):

        log_path = r"yloop{}_tSize{}_xy{}{}_{}_BDA{}_ws{}_eta{}_exp{}_c{}_etaCG{}_ew{}_le{}._convex_{}.mat".format(args.y_loop, args.tSize,
                                                                                                x0, y0, args.hg_mode,
                                                                                                args.BDA,
                                                                                                args.WarmStart,
                                                                                                args.eta, args.exprate,
                                                                                                args.c,args.eta_CG,args.element_wise,args.lin_error, time.strftime(
                "%Y_%m_%d_%H_%M_%S"),
                                                                                                )
        d = 28 ** 2
        n = 10
        z_loop = args.z_loop
        y_loop = args.y_loop
        x_loop = args.x_loop

        val_losses = []

        a_tensor = create_tensor_with_xy(args.tSize,(1.0/args.tSize),(-1.0/args.tSize)).requires_grad_(False)


        class ModelTensorF(torch.nn.Module):
            def __init__(self, tensor):
                super(ModelTensorF, self).__init__()
                self.p = torch.nn.Parameter(tensor)

            def forward(self):
                return self.p

            def t(self):
                return self.p.t()


        class ModelTensorf(torch.nn.Module):
            def __init__(self, tensor):
                super(ModelTensorf, self).__init__()
                self.p = torch.nn.Parameter(tensor)

            def forward(self):
                return self.p

            def t(self):
                return self.p.t()


        tSize = args.tSize

        Fmin = 0
        xmin = []
        xmind = 0
        for i in range(tSize):
            Fmin = Fmin + (-np.pi / 4 / (i + 1) - a) ** 2 + (-np.pi / 4 / (i + 1) - b) ** 2
            xmin.append(-np.pi / 4 / (i + 1))
            xmind = xmind + (-np.pi / 4 / (i + 1)) ** 2

        x = ModelTensorF( create_tensor_with_xy(args.tSize,1,1)).requires_grad_(True)
        y = ModelTensorF( create_tensor_with_xy(args.tSize,1,1)).requires_grad_(True)


        C = (float(2) * torch.ones(tSize)).cuda().requires_grad_(False)
        e = torch.ones(tSize).requires_grad_(False)


        def val_loss(params,hparams=0):
            val_ = 0
            for sz in range(args.tSize):
                # print(fmodel(params=params))
                val_  += fmodel(params=params)[sz]
            return val_


        inner_losses = []
        def train_loss(params, hparams=0):
            train_loss_f = 0.5 * torch.sum(torch.square(fmodel(params=params)-a_tensor))
            reg_ = 0
            for sz in range(args.tSize):
                reg_  += x()[sz]* torch.norm(fmodel(params=params)[sz], 1)
            train_loss = train_loss_f + reg_
            #              + x()[0]* torch.norm(fmodel(params=params), 1) + 0.5 * \
            # x()[1] * torch.sum(torch.square(fmodel(params=params)))/100
            return train_loss


        def train_loss_f(params, hparams=0):
            # print(np.shape(labels))
            # print(np.shape(torch.mm(inputs,fmodel(params=params))))
            train_loss_f = 0.5 * torch.sum(torch.square(fmodel(params=params)-a_tensor))

            return train_loss_f

        def low_loss_FO(theta, params, hparams, ck, gamma):
            # vs_tensor = torch.tensor([item.cpu().detach().numpy() for item in theta]).cuda()
            # param_tensor = torch.tensor([item.cpu().detach().numpy() for item in params]).cuda()
            result_params = []

            reg = 0
            for param1, param2 in zip(theta, params):
                diff = param1 - param2
                reg += torch.norm(diff)**2
            return ck*val_loss(params, hparams) + train_loss_f(params, hparams) \
                   - 0.5 * gamma * reg



        def upper_loss_FO(theta,params,hparams,ck):
            return ck*val_loss(params,hparams) + train_loss(params,hparams) - train_loss(theta,hparams)

        def train_loss_error(params, theta, hparams=0):
            out=0.5*torch.norm(fmodel(1,params=params))**2-sum(x(-1)*fmodel(1,params=params))
            out_2 = 0.5*torch.norm(fmodel(1,params=theta))**2-sum(x(-1)*fmodel(1,params=theta))
            return out - out_2



        inner_losses = []


        def train_loss_BDA(params, hparams=0, alpha=0.1):
            out = (1 - alpha) * train_loss(params) + alpha * val_loss(params)
            return out


        def inner_loop(hparams, params, optim, n_steps, log_interval, create_graph=False):
            params_history = [optim.get_opt_params(params)]

            for t in range(n_steps):
                params_history.append(optim(params_history[-1], hparams, create_graph=True))

                # if log_interval and (t % log_interval == 0 or t == n_steps-1):
                #     print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

            return params_history



        def inner_loop2(loss,hparams, params, optim, n_steps, log_interval, create_graph=False):
            params_history = [optim.get_opt_params(params)]

            for t in range(n_steps):
                params_history.append(optim(loss,params_history[-1], hparams, create_graph=True))


            return params_history

        def inner_loop_CG(hparams, params, optim, n_steps, log_interval, create_graph=False):
            params_history = [optim.get_opt_params(params)]

            for t in range(n_steps):
                params_history = [(optim(params_history[-1], hparams))]

            return params_history


        def soft_thresholding(input_tensor, threshold):

            return torch.sign(input_tensor) * torch.relu(torch.abs(input_tensor) - threshold)


        def apply_soft_thresholding_to_list(tensor_list, threshold):

            processed_tensors = [soft_thresholding(tensor, threshold) for tensor in tensor_list]
            return processed_tensors
        x_opt = torch.optim.SGD(x.parameters(), lr=args.x_lr)
        y_opt = torch.optim.SGD(y.parameters(), lr=args.y_lr)

        acc_history = []
        clean_acc_history = []

        loss_x_l = 0
        F1_score_last = 0
        lr_decay_rate = 1
        reg_decay_rate = 1
        dc = 0
        total_time = 0
        total_hyper_time = 0
        v = -1
        eta = args.eta * args.c
        if if_cuda:
            x=x.cuda()
            y=y.cuda()
            e=e.cuda()
            xh=xh.cuda()
            xstar=xstar.cuda()

        for x_itr in range(x_loop):
            x_opt.zero_grad()
            if args.linear:
                eta = args.eta - (args.eta * (x_itr + 1) / x_loop)
            if args.exp:
                eta = eta * args.exprate
            if x_itr > 100:
                args.GN = False

            if args.thelr:
                x_lr=args.x_lr*(x_itr+1)**(-args.tau)*args.y_lr
                eta=args.eta0*(x_itr+1)**(-0.5*args.tau)*args.y_lr

                for params in x_opt.param_groups:
                    params['lr'] =  x_lr

            if args.hg_mode == 'MEHA':
                eta = 0.5
                gamma_1 = 0.01
                c0 =0.5
                if x_itr==0:
                    ck = np.power(x_itr+1,0.49)*c0
                else:
                    ck = np.power(x_itr+1,0.49)*c0
                t0 = time.time()
                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)
                params = [w.detach().requires_grad_(True) for w in list(y.parameters())]  # y
                vs = [torch.zeros_like(w).requires_grad_(True) for w in params] if v == -1 else v
                theta_loss = train_loss_f(params=vs)
                grad_theta_parmaters = grad_unused_zero(theta_loss,vs)
                t1 = time.time()
                errs = []
                for a, b in zip(vs, params):
                    diff = a - b
                    errs.append(diff)
                t2 = time.time()
                vs = [v0 - eta * (gt +gamma_1 * err) for v0, gt, err in
                      zip(vs, grad_theta_parmaters,  errs)]
                vs_ = apply_soft_thresholding_to_list(vs,eta)
                t3 = time.time()
                # y_opt.zero_grad()
                lower_loss  = low_loss_FO(vs,list(y.parameters()),x.parameters(),ck,gamma_1)
                grad_y_parmaters = grad_unused_zero(lower_loss,list(y.parameters()))
                update_tensor_grads(y.parameters(),grad_y_parmaters)
                y_opt.step()
                y_list = apply_soft_thresholding_to_list(list(y.parameters()),1e-2)
                copy_parameter_from_list(y, y_list)
                params = [w.detach().requires_grad_(True) for w in list(y.parameters())]  # y
                upper_loss = upper_loss_FO(vs,params,x.parameters(),ck)
                grad_x_parmaters = grad_unused_zero(upper_loss,list(x.parameters()))
                update_tensor_grads(list(x.parameters()), grad_x_parmaters)
            else:
                print('NO hypergradient!')


            xgrad=[x0g.grad for x0g in x.parameters()]

            x_opt.step()
            min_value = 0.0
            max_value = 1.0
            for param in x.parameters():
                param.data = torch.clamp(param.data, min_value, max_value)
            step_time = time.time() - t0
            total_time += time.time() - t0
            test_error_x = torch.norm((x() - xstar)).detach().cpu().numpy()
            test_error_y = torch.norm((y() - ystar)).detach().cpu().numpy()
            if test_error_x<=0.02 and test_error_y<0.1:
                print(total_time,'total_time',test_error_x,'test_x',test_error_y,'test_y', (x_itr+1), 'x_loop')
                break
            if (x_itr+1) % args.log == 0:

                print(total_time,'total_time',test_error_x,'test_x',test_error_y,'test_y', (x_itr+1), 'x_loop')
                time_list.append(total_time)

scipy.io.savemat('./non_smooth'+str(args.tSize)+'.mat', mdict={'x': x_error_list, 'y': y_error_list,  'time': time_list,
                                    })