import torch
import copy
import numpy as np
import time
import csv
import argparse
import sys
sys.path.append("..")
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

parser = argparse.ArgumentParser(description='merely convex')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='FashionMNIST', metavar='N')
parser.add_argument('--mode', type=str, default='ours', help='ours or IFT')
parser.add_argument('--hg_mode', type=str, default='MEHA', metavar='N',
                    help='BAMM_RHG, RHG, CG, BDA, fixed_point')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--z_loop', type=int, default=10)
parser.add_argument('--y_loop', type=int, default=1)
parser.add_argument('--x_loop', type=int, default=725)
parser.add_argument('--z_L2_reg', type=float, default=0.01)
parser.add_argument('--y_L2_reg', type=float, default=0.01)
parser.add_argument('--y_ln_reg', type=float, default=0.1)
parser.add_argument('--reg_decay', type=bool, default=True)
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--learn_h', type=bool, default=False)
parser.add_argument('--x_lr', type=float, default=0.01)
parser.add_argument('--y_lr', type=float, default=0.1)
parser.add_argument('--z_lr', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--p0', type=float, default=10)
parser.add_argument('--alpha0', type=float, default=0.1)
parser.add_argument('--beta0', type=float, default=0.1)
parser.add_argument('--eta0', type=float, default=0.5)
parser.add_argument('--mu0', type=float, default=0.9)
parser.add_argument('--x_lr_decay_rate', type=float, default=0.1)
parser.add_argument('--x_lr_decay_patience', type=int, default=1)
parser.add_argument('--pollute_rate', type=float, default=0.5)
parser.add_argument('--convex', type=bool, default=True)
parser.add_argument('--tSize', type=int, default=10)
parser.add_argument('--BDA', action='store_true',default=False)
parser.add_argument('-ew','--element_wise', action='store_true', default=False)
parser.add_argument('--linear', action='store_true', default=False)
parser.add_argument('--exp', action='store_true', default=False)
parser.add_argument('--thelr', action='store_true', default=False)# whether to use standerd strategy for update

parser.add_argument('--eta_CG', type=int, default=-1)

parser.add_argument('--c', type=float, default=2.)
parser.add_argument('--tau', type=float, default=1/40.)

parser.add_argument('--exprate', type=float, default=1.)

parser.add_argument('-ws', '--WarmStart', action='store_false', default=True)
parser.add_argument('--eta', type=float, default=1.)
args = parser.parse_args()
if not args.WarmStart or args.hg_mode.find('BAMM')!=-1:
    print('One Stage')
    # args.x_loop= args.x_loop* args.y_loop
    args.y_loop=1

print(args)

def show_memory_info(hint):
    pid = os.getpid()

    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss/1024./1024
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


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


class Net_x(torch.nn.Module):
    def __init__(self, tr):
        super(Net_x, self).__init__()
        self.x = torch.nn.Parameter(torch.zeros(tr.data.shape[0]).to("cuda").requires_grad_(True))

    def forward(self, y):
        # if torch.norm(torch.sigmoid(self.x), 1) > 2500:
        #     y = torch.sigmoid(self.x) / torch.norm(torch.sigmoid(self.x), 1) * 2500 * y
        # else:
        y = torch.sigmoid(self.x) * y
        y = y.mean()
        return y


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

    for p, q in zip(y.parameters(), z.parameters()):
        p.data = q.clone().detach().requires_grad_()

    return y


def copy_parameter_from_list(y, z):

    for p, q in zip(y.parameters(), z):
        p.data = q.clone().detach().requires_grad_()

    return y

gradlist=[]
xlist = []
ylist = []
zlist = []
Flist=[]
etalist=[]
xgradlist=[]
alphalist = []
timelist = []
xstar=torch.ones([args.tSize,1]).cuda()
ystar=torch.ones([args.tSize,1]).cuda()
eta=args.c*args.eta
args.x_lr0 =args.x_lr
for x0,y0 in zip([0.01],[0.01]):
    for a,b in zip([2],[2]):

        log_path = "yloop{}_tSize{}_xy{}{}_{}_BDA{}_{}_convex_{}.mat".format(args.y_loop,args.tSize,x0,y0,args.hg_mode,args.BDA,args.c, time.strftime("%Y_%m_%d_%H_%M_%S"),
                                                                   )
        with open(log_path, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow( ['y_loop{}y_lr{}x_lr{}'.format(args.y_loop,args.y_lr, args.x_lr),
                         'time', 'x','hyper_time', 'lower_time','y'])
        d = 28 ** 2
        n = 10
        z_loop = args.z_loop
        y_loop = args.y_loop
        x_loop = args.x_loop

        val_losses = []


        class ModelTensorF(torch.nn.Module):
            def __init__(self, tensor):
                super(ModelTensorF, self).__init__()
                self.T = torch.nn.Parameter(tensor)

            def forward(self,i=-1):
                if i==-1:
                    return self.T
                else:
                    return self.T[i]


        class ModelTensorf(torch.nn.Module):
            def __init__(self, tensor):
                super(ModelTensorf, self).__init__()
                self.T1 = torch.nn.Parameter(tensor)
                self.T2 = torch.nn.Parameter(tensor)

            def forward(self, i=1):
                if i == 1:
                    return self.T1
                else:
                    return self.T2

        tSize=args.tSize

        x = ModelTensorF((float(x0)*torch.ones(tSize)).cuda().requires_grad_(True))
        y = ModelTensorf((float(y0)*torch.ones(tSize)).cuda().requires_grad_(True))

        C=(float(2)*torch.ones(tSize)).cuda().requires_grad_(False)
        e=torch.ones(tSize).cuda().requires_grad_(False)


        def val_loss(params, hparams=0):

            val_loss =0.5* torch.norm(x()-fmodel(2,params=params))**2 + 0.5*torch.norm(fmodel(1,params=params)-e)**2

            return val_loss

        inner_losses = []


        def train_loss(params, hparams=0):
            out=0.5*torch.norm(fmodel(1,params=params))**2-sum(x(-1)*fmodel(1,params=params))
            return out


        def low_loss_FO(theta, params, hparams, ck, gamma):
            # vs_tensor = torch.tensor([item.cpu().detach().numpy() for item in theta]).cuda()
            # param_tensor = torch.tensor([item.cpu().detach().numpy() for item in params]).cuda()
            result_params = []

            reg = 0
            for param1, param2 in zip(theta, params):
                diff = param1 - param2
                reg += torch.norm(diff)**2
            return ck*val_loss(params, hparams) + train_loss(params, hparams) - 0.5 * gamma * reg



        def upper_loss_FO(theta,params,hparams,ck):
            return ck*val_loss(params,hparams) + train_loss(params,hparams) - train_loss(theta,hparams)

        def train_loss_error(params, theta, hparams=0):
            out=0.5*torch.norm(fmodel(1,params=params))**2-sum(x(-1)*fmodel(1,params=params))
            out_2 = 0.5*torch.norm(fmodel(1,params=theta))**2-sum(x(-1)*fmodel(1,params=theta))
            return out - out_2



        inner_losses = []


        def train_loss_BDA(params, hparams=0,alpha = args.alpha):

            out=(1-alpha)*train_loss(params)+alpha*val_loss(params)
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
                params_history=[(optim(params_history[-1], hparams))]
            return params_history


        x_opt = torch.optim.Adam(x.parameters(), lr=args.x_lr)
        y_opt = torch.optim.SGD(y.parameters(), lr=args.y_lr, momentum=0.1)

        acc_history = []
        clean_acc_history = []

        loss_x_l = 0
        F1_score_last = 0
        lr_decay_rate = 1
        reg_decay_rate = 1
        dc = 0
        total_time = 0
        total_hyper_time = 0
        #BAMM
        # v = -1
        v= -1
        # show_memory_info(1)
        args.y_lr0 = args.y_lr

        for x_itr in range(x_loop):
            args.alpha=args.alpha*(x_itr+1)/(x_itr+2)
            if args.linear:
                eta = args.eta - (args.eta * (x_itr + 1) / x_loop)
            if args.exp:
                eta = eta * args.exprate
            if x_itr > 100:
                args.GN = False

            # if args.thelr:
            #     args.alpha=args.mu0*1/(x_itr+1)**(1/args.p0)
            #     eta = (x_itr+1)**(-0.5 * args.tau) * args.y_lr
            #     args.x_lr=(x_itr+1)**(-1.5 * args.tau)*args.alpha**3*args.y_lr
            #
            # if args.thelr:
            #     args.alpha = args.mu0 * 1 / (x_itr + 1) ** (1 / args.p0)
            #     eta = (x_itr + 1) ** (-0.5 * args.tau) * args.y_lr
            #     args.x_lr = (x_itr + 1) ** (-1.5 * args.tau) * args.alpha ** 3 * args.y_lr
            #
            #     for params in x_opt.param_groups:
            #         params['lr'] =  args.x_lr
            #     for params in y_opt.param_groups:
            #         params['lr'] = args.y_lr

            a= torch.any(torch.isnan(x()))
            if  torch.any(torch.isnan(x())):
                break
            x_opt.zero_grad()

            args.x_lr =args.x_lr0*1.2
            c0 = 0.6
            if args.hg_mode == 'MEHA':
                ### hyper-parameters
                eta = 0.009
                gamma_1 = 0.2
                if x_itr==0:
                    ck = np.power(x_itr+1,0.49)*c0
                else:
                    ck = np.power(x_itr+1,0.49)*c0
                t0 = time.time()
                fmodel = higher.monkeypatch(y, torch.device("cuda"), copy_initial_weights=True)

                #### Update \theta We assuptation v
                params = [w.detach().requires_grad_(True) for w in list(y.parameters())]  # y
                vs = [torch.zeros_like(w).requires_grad_(True) for w in params] if v == -1 else v
                theta_loss = train_loss(params=vs,hparams=x.parameters())
                grad_theta_parmaters = grad_unused_zero(theta_loss,vs)
                t1 = time.time()
                errs = []
                for a, b in zip(vs, params):
                    diff = a - b
                    errs.append(diff)

                t2 = time.time()

                vs = [v0 - eta * (gt +gamma_1 * err) for v0, gt, err in
                      zip(vs, grad_theta_parmaters,  errs)]  # upate \theta

                t3 = time.time()


                inner_opt = hg.GradientDescent2(low_loss_FO, step_size=args.y_lr)
                lower_loss  = low_loss_FO(vs,list(y.parameters()),x.parameters(),ck,gamma_1)
                last_param = inner_loop2(lower_loss,x.parameters(), y.parameters(), inner_opt, 1, log_interval=10)
                copy_parameter_from_list(y, last_param[-1])
                params = [w.detach().requires_grad_(True) for w in list(y.parameters())]  # y

                upper_loss = upper_loss_FO(vs,params,list(x.parameters()),ck)
                grad_x_parmaters = grad_unused_zero(upper_loss,list(x.parameters()))

                # update_tensor_grads(list(x.parameters()), grad_x_parmaters)

                #
                # def gd_step(params, loss, step_size, create_graph=True):
                #     grads = torch.autograd.grad(loss, params, create_graph=create_graph, allow_unused=True)

                params_hyper = [w - args.x_lr * (g if g is not None else 0) for w, g in zip(list(x.parameters()), grad_x_parmaters)]
                copy_parameter_from_list(x, params_hyper)

                t4 = time.time()

                x_time = time.time() -t0
                print(x_time)
                v = vs


                print('newmethod', x_time)


            else:
                print('NO hypergradient!')
            xgrad=[x0g for x0g in grad_x_parmaters]

            if 'BAMM' in args.hg_mode:

                time_list=[y_time,prepare_time+x_time,prepare_time+v_time]
                outer_time=max(time_list)
                step_time = outer_time
                total_hyper_time += prepare_time+x_time+v_time
                total_time += outer_time
            else:

                step_time=time.time() - t0
                # total_hyper_time += time.time() - hyper_time
                total_time += time.time() - t0

            if x_itr % 1 == 0:

                xgradlist.append(copy.deepcopy([x0.detach().cpu().numpy() for x0 in xgrad]))

                with torch.no_grad():
                    with torch.no_grad():

                        print(
                            'x_itr={},xdist={:.6f},ydist={:.6f},zdist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                                x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                                (torch.norm((y(1) - ystar)) / torch.norm(ystar)).detach().cpu().numpy(),
                                (torch.norm((y(2) - ystar)) / torch.norm(ystar)).detach().cpu().numpy(),
                                xgradlist[-1][0][0],
                                total_hyper_time, total_time))
                        xlist.append(copy.deepcopy(x().detach().cpu().numpy()))
                        ylist.append(y().detach().cpu().numpy())
                        zlist.append(copy.deepcopy(y(2).detach().cpu().numpy()))

                        timelist.append(total_time)
                        alphalist.append(args.alpha)
                        print(xlist[-1])

            if torch.norm((x() - xstar) / xstar).detach().cpu().numpy()<1e-2:#timelist[-1]>15:# loss_L2(xgrad) < 1e-6:
                print(
                    'x_itr={},xdist={:.6f},ydist={:.6f},zdist={:.6f},xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                        x_itr, torch.norm((x() - xstar) / xstar).detach().cpu().numpy(),
                        (torch.norm((y(1) - ystar)) /torch.norm (ystar)).detach().cpu().numpy(),
                        (torch.norm((y(2) - ystar)) / torch.norm(ystar)).detach().cpu().numpy(),

                        xgradlist[-1][0][0],
                        total_hyper_time, total_time))
                break
                xgradlist.append(copy.deepcopy([x0.detach().cpu().numpy() for x0 in xgrad]))

scipy.io.savemat(log_path, mdict={'x': xlist, 'y': ylist, 'z': zlist, 'time': timelist, 'n': args.tSize,
                                                  'xstar': xstar.cpu().numpy(), 'F': Flist, 'xgrad': xgradlist,
                                                  'eta': etalist,'alpha':alphalist, 'Memory': torch.cuda.max_memory_allocated()})