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
parser.add_argument('--y_L2_reg', type=float, default=0.02)
parser.add_argument('--y_ln_reg', type=float, default=0.1)
parser.add_argument('--reg_decay', type=bool, default=True)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--learn_h', type=bool, default=False)
parser.add_argument('--x_lr', type=float, default=0.0005)
parser.add_argument('--y_lr', type=float, default=0.0005)
parser.add_argument('--z_lr', type=float, default=0.01)
parser.add_argument('--x_lr_decay_rate', type=float, default=0.1)
parser.add_argument('--x_lr_decay_patience', type=int, default=1)
parser.add_argument('--pollute_rate', type=float, default=0.5)
parser.add_argument('--convex', type=bool, default=True)
parser.add_argument('--tSize', type=int, default=1)
parser.add_argument('--xSize', type=int, default=1)
parser.add_argument('--ySize', type=int, default=1)
parser.add_argument('--log', type=int, default=5)#10
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


a_parameter = float(2.0)*torch.ones(args.xSize).requires_grad_(False)
c_parameter = float(2.0)*torch.ones(args.ySize).requires_grad_(False)



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
    pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
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
        self.toy_x = torch.nn.Parameter(args.x0 * torch.ones(dx).requires_grad_(True))

    def forward(self, y, b ,c , x=None):
        # Ba = b @ self.a
        if x is None:
            # if args.hg_mode=='BDAn' or args.hg_mode =='BDA':
            #     return  self.toy_x + self.toy_x * lim(y,-2,2)
            # else:
            return torch.norm(self.toy_x - self.a) ** 2 + torch.norm(y - self.a - c) ** 2
            # return  self.toy_x + self.toy_x * y
        else:
            return torch.norm(x - self.a) ** 2 + torch.norm(y - self.a - c) ** 2


def copy_parameter(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z.parameters()):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y


def frnp(x):
    t = torch.from_numpy(x).cuda()
    return t


def tonp(x):
    return x.detach().cpu().numpy()


def copy_parameter(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z.parameters()):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y


def copy_parameter_from_list(y, z):
    # print(loss_L1(y.parameters()))
    # print(loss_L1(z.parameters()))
    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()
    # print(loss_L1(y.parameters()))
    # print('-'*80)
    return y

import math

idt = args.idt
if idt:

    A= 1
    B = 1
    D = 1
    invA = A

    xh = torch.ones([args.xSize, 1]) * 1
    ystar = 0.75*math.pi+2
    BinvATBT = 1
    inv_BinvATBT_mD = 0.5
    xstar = 0.75*math.pi
else:
    A, invA = positive_matrix(args.ySize)
    B = torch.rand([args.xSize, args.ySize])
    D, invD = semipositive_matrix(args.xSize)
    xh = torch.rand([args.xSize, 1]) * 1
    ystar = torch.mm(invA, B.t())
    BinvATBT = torch.mm(torch.mm(B, invA.t()), B.t())
    inv_BinvATBT_mD = torch.inverse(BinvATBT + D)
    xstar = torch.mm(inv_BinvATBT_mD, torch.mm(D, xh))

# x0=float(args.x0)
# y0=float(args.y0)
gradlist = []
xlist = []
ylist = []
vlist = []
Flist=[]
etalist=[]
xgradlist=[]
timelist = []

for x0, y0 in zip([-6], [-0]):
    for a, b in zip([2], [2]):

        log_path = r"yloop{}_tSize{}_xy{}{}_{}_BDA{}_ws{}_eta{}_exp{}_c{}_etaCG{}_ew{}_le{}._convex_{}.mat".format(args.y_loop, args.tSize,
                                                                                                x0, y0, args.hg_mode,
                                                                                                args.BDA,
                                                                                                args.WarmStart,
                                                                                                args.eta, args.exprate,
                                                                                                args.c,args.eta_CG,args.element_wise,args.lin_error, time.strftime(
                "%Y_%m_%d_%H_%M_%S"),
                                                                                                )
        # with open(log_path, 'a', encoding='utf-8') as f:
        #     csv_writer = csv.writer(f)
        #     # csv_writer.writerow([args])
        #     csv_writer.writerow( ['y_loop{}y_lr{}x_lr{}'.format(args.y_loop,args.y_lr, args.x_lr),
        #                  'time', 'x','hyper_time', 'lower_time','y'])
        d = 28 ** 2
        n = 10
        z_loop = args.z_loop
        y_loop = args.y_loop
        x_loop = args.x_loop

        val_losses = []


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

        x = ModelTensorF(float(x0) * torch.ones(args.ySize)).requires_grad_(True)
        y = ModelTensorF(float(y0) * torch.ones(args.ySize)).requires_grad_(True)


        C = (float(2) * torch.ones(tSize)).cuda().requires_grad_(False)
        e = torch.ones(tSize).requires_grad_(False)


        def val_loss(params,b,c, hparams):
            # if x is None:
            #     # if args.hg_mode=='BDAn' or args.hg_mode =='BDA':
            #     #     return  self.toy_x + self.toy_x * lim(y,-2,2)
            #     # else:
            #     return torch.norm(hparams - a_parameter) ** 2 + torch.norm(params - hparams - c) ** 2
            #     # return  self.toy_x + self.toy_x * y
            # else:
            return torch.norm(hparams[0] - a_parameter) ** 2 + torch.norm(params[0] - a_parameter - c) ** 2



        inner_losses = []


        def train_loss(params,b, hparams):
            out = 0
            for i in range(args.ySize):
                out = out + torch.sin((hparams[i] + params[i] - c_parameter[i]))
            return out


        def low_loss_FO(theta, params, hparams, ck, gamma,b,c):
            # vs_tensor = torch.tensor([item.cpu().detach().numpy() for item in theta]).cuda()
            # param_tensor = torch.tensor([item.cpu().detach().numpy() for item in params]).cuda()
            result_params = []


            reg = 0
            for param1, param2 in zip(theta, params):
                diff = param1 - param2
                # result_params.append(diff)
                reg += torch.norm(diff)**2
            return ck*val_loss(params,b,c, hparams) + train_loss(params,b, hparams) - 0.5 * gamma * reg



        def upper_loss_FO(theta,params,hparams,ck,b,c):
            return ck*val_loss(params,b,c,hparams) + train_loss(params,b, hparams) - train_loss(theta,b,hparams)

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


        def lim(x, min, max):
            return torch.minimum(torch.maximum(x, torch.ones_like(x) * min), torch.ones_like(x) * max)

        x_opt = torch.optim.SGD(x.parameters(), lr=args.x_lr)
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
        B = torch.nn.Parameter(torch.ones(args.ySize, args.ySize)).requires_grad_(False)

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
                eta = 0.001
                gamma_1 = 0.005
                c0 =50

                if x_itr == 0:
                    ck = np.power(x_itr + 1, 0.48) * c0
                else:
                    ck = np.power(x_itr + 1, 0.48) * c0

                t0 = time.time()
                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)

                params = [w.detach().requires_grad_(True) for w in list(y.parameters())]  # y
                vs = [torch.zeros_like(w).requires_grad_(True) for w in params] if v == -1 else v
                theta_loss = train_loss(params=vs,b=B,hparams=list(x.parameters()))

                grad_theta_parmaters = grad_unused_zero(theta_loss,vs)

                t1 = time.time()

                errs = []
                for a, b in zip(vs, params):
                    diff = a - b
                    errs.append(diff)

                t2 = time.time()
                if params[0] - ystar >= 0.1:
                    vs = [v0 - eta * (gt +gamma_1 * err) for v0, gt, err in
                          zip(vs, grad_theta_parmaters,  errs)]  # upate \theta

                t3 = time.time()
                params = [w.detach().requires_grad_(True) for w in list(y.parameters())]  # y

                upper_loss = upper_loss_FO(vs, params, list(x.parameters()), ck, B, c_parameter)
                grad_x_parmaters = grad_unused_zero(upper_loss, list(x.parameters()))

                update_tensor_grads(list(x.parameters()), grad_x_parmaters)
                x_opt.step()

                inner_opt = hg.GradientDescent2(low_loss_FO, step_size=args.y_lr)
                lower_loss  = low_loss_FO(vs,list(y.parameters()),list(x.parameters()),ck,gamma_1, B, c_parameter)
                last_param = inner_loop2(lower_loss,list(x.parameters()), list(y.parameters()), inner_opt, 1, log_interval=10)
                last_param_new = lim(last_param[-1][0], -10, 10)
                if last_param_new-ystar>=0.1:
                    copy_parameter_from_list(y, [last_param_new])
                t4 = time.time()

                x_time = time.time() -t0
                v = vs


            else:
                print('NO hypergradient!')

            xgrad=[x0g.grad for x0g in x.parameters()]
            print(len(xgrad))
            copy_parameter_from_list(y, last_param[-1])
            step_time = time.time() - t0
            total_time += time.time() - t0

            if x_itr % args.log == 0:

                xgradlist.append(copy.deepcopy([x0.detach().cpu().numpy() for x0 in xgrad]))


                with torch.no_grad():
                    with torch.no_grad():
                        print(
                            'x_itr={},xdist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                                x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                                xgradlist[-1][0][0],
                                total_hyper_time, total_time))
                        xlist.append(copy.deepcopy(x().detach().cpu().numpy()))
                        ylist.append(y().detach().cpu().numpy())
                        etalist.append(eta.detach().cpu().numpy() if torch.is_tensor(eta) else eta)
                        if v == -1:
                            vlist.append(v)
                        else:
                            vlist.append([v0.detach().cpu().numpy() for v0 in v])
                        timelist.append(total_time)
                        print(val_loss(last_param[-1],B, c_parameter,x()))
                        Flist.append(val_loss(last_param[-1],B, c_parameter,x()).detach().cpu().numpy())
                        if len(xlist) > 1:
                            print(loss_L2(xgrad))
            if len(xlist)>1:
                if loss_L2(xgrad)<=1e-9:#timelist[-1]>30 or args.element_wise:# or args.element_wise:#timelist[-1]>20loss_L2(xgrad)<1e-8:#loss_L2(xgrad)<1e-4:#timelist[-1]>6:#loss_L2(xgrad)<1e-8:#np.linalg.norm(xlist[-1]-xlist[-2])/np.linalg.norm(xlist[-2])<1e-8:
                    print(
                        'x_itr={},xdist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                            x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                            xgradlist[-1][0][0],
                            total_hyper_time, total_time))
                    xgradlist.append(copy.deepcopy([x0.detach().cpu().numpy() for x0 in xgrad]))
                    Flist.append(val_loss(last_param[-1], B, c_parameter, x()).detach().cpu().numpy())

                    with torch.no_grad():
                        with torch.no_grad():
                            xlist.append(copy.deepcopy(x().detach().cpu().numpy()))
                            ylist.append(y().detach().cpu().numpy())
                            etalist.append(eta.detach().cpu().numpy() if torch.is_tensor(eta) else eta)
                            if v == -1:
                                vlist.append(v)
                            else:
                                vlist.append([v0.detach().cpu().numpy() for v0 in v])
                            timelist.append(total_time)
                    break

if args.idt:
    scipy.io.savemat(log_path, mdict={'x': xlist, 'y': ylist, 'v': vlist, 'time': timelist,'n':args.xSize,
                                   'xh': xh.cpu().numpy(),
                                  'xstar': xstar,'F':Flist,'xgrad':xgradlist,'eta':etalist,'Memory':torch.cuda.max_memory_allocated()})
else:
    scipy.io.savemat(log_path, mdict={'x': xlist, 'y': ylist, 'v': vlist, 'time': timelist, 'A': A.cpu().numpy(),
                                  'B': B.cpu().numpy(), 'D': D.cpu().numpy(), 'xh': xh.cpu().numpy(),
                                  'xstar': xstar.cpu().numpy(),'F':Flist,'xgrad':xgradlist,'eta':etalist,'Memory':torch.cuda.max_memory_allocated()})
