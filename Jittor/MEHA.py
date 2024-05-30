import jittor as jt
import numpy as np
import argparse
import time
import csv
import copy
import scipy.io


def grad_unused_zero(output, inputs, retain_graph=False, create_graph=False):
    grads = jt.grad(output, inputs, retain_graph=retain_graph)

    def grad_or_zeros(grad, var):
        return jt.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


def copy_parameter_from_list(y, z):
    for p, q in zip(y.parameters(), z):
        p.update(q.clone().detach())

    return y
# 解析命令行参数
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

# 确保Jittor使用GPU
gradlist=[]
xlist = []
ylist = []
zlist = []
Flist=[]
etalist=[]
xgradlist=[]
alphalist = []
timelist = []
xstar = jt.ones(args.tSize).to(jt.float32)
ystar = jt.ones(args.tSize).to(jt.float32)
eta = args.c * args.eta

jt.flags.use_cuda = 1
args.x_lr0 =args.x_lr

for x0, y0 in zip([0.01], [0.01]):
    for a, b in zip([2], [2]):
        log_path = "yloop{}_tSize{}_xy{}{}_{}_BDA{}_{}_convex_{}.mat".format(args.y_loop, args.tSize, x0, y0,
                                                                             args.hg_mode, args.BDA, args.c,
                                                                             time.strftime("%Y_%m_%d_%H_%M_%S"))

        with open(log_path, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                ['y_loop{}y_lr{}x_lr{}'.format(args.y_loop, args.y_lr, args.x_lr), 'time', 'x', 'hyper_time',
                 'lower_time', 'y'])

        d = 28 ** 2
        n = 10
        z_loop = args.z_loop
        y_loop = args.y_loop
        x_loop = args.x_loop

        val_losses = []


        class ModelTensorF(jt.nn.Module):
            def __init__(self, tensor):
                super(ModelTensorF, self).__init__()
                self.T = jt.nn.Parameter(tensor)

            def execute(self, i=-1):
                if i == -1:
                    return self.T
                else:
                    return self.T[i]


        class ModelTensorf(jt.nn.Module):
            def __init__(self, tensor):
                super(ModelTensorf, self).__init__()
                self.T1 = jt.nn.Parameter(tensor)
                self.T2 = jt.nn.Parameter(tensor)

            def execute(self, i=1):
                if i == 1:
                    return self.T1
                else:
                    return self.T2


        tSize = args.tSize

        x = ModelTensorF((float(x0) * jt.ones(tSize)))
        y = ModelTensorf((float(y0) * jt.ones(tSize)))
        theta_model = ModelTensorf((float(y0) * jt.ones(tSize)))

        C = (float(2) * jt.ones(tSize)).stop_grad()
        e = jt.ones(tSize).stop_grad()

        def val_loss(x,y):

            val_loss = 0.5 * jt.norm(x(-1) - y(2)) ** 2 + 0.5 * jt.norm(y(1) - e) ** 2
            return val_loss

        inner_losses = []
        def train_loss(y):
            out = 0.5 * jt.norm(y(1)) ** 2 - jt.sum(x(-1) * y(1))
            return out


        def low_loss_FO(theta, y, x, ck, gamma):
            result_params = []
            reg = 0
            for param1, param2 in zip(theta.parameters(), y.parameters()):
                diff = param1 - param2
                reg += jt.norm(diff) ** 2
            return ck * val_loss(x,y) + train_loss(y) - 0.5 * gamma * reg
        def upper_loss_FO(theta, y, x, ck):
            return ck * val_loss(y, x) + train_loss(y) - train_loss(theta)





        x_opt = jt.optim.SGD(x.parameters(), lr=args.x_lr)
        y_opt = jt.optim.SGD(y.parameters(), lr=args.y_lr, momentum=0.1)

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
        args.y_lr0 = args.y_lr

        for x_itr in range(x_loop):
            args.alpha = args.alpha * (x_itr + 1) / (x_itr + 2)
            if args.linear:
                eta = args.eta - (args.eta * (x_itr + 1) / x_loop)
            if args.exp:
                eta = eta * args.exprate
            if x_itr > 100:
                args.GN = False

            a = jt.any(jt.isnan(x()))
            if jt.any(jt.isnan(x())):
                break
            x_opt.zero_grad()

            args.x_lr = args.x_lr0 * 1.2
            c0 = 0.6
            if args.hg_mode == 'MEHA':
                eta = 0.009
                gamma_1 = 0.2
                if x_itr == 0:
                    ck = np.power(x_itr + 1, 0.49) * c0
                else:
                    ck = np.power(x_itr + 1, 0.49) * c0
                t0 = time.time()
                # fmodel = higher.monkeypatch(y, jt.flags.use_cuda, copy_initial_weights=True)

                params = [w.detach().stop_grad() for w in list(y.parameters())]
                vs = [jt.zeros_like(w).stop_grad() for w in params] if v == -1 else v
                for param in y.parameters():
                    param.start_grad()
                params_theta = list(theta_model.parameters())
                for param,v in zip(params_theta,vs):
                    param.update(v)
                for param in params_theta:
                    param.start_grad()
                theta_loss = train_loss(theta_model)
                grad_theta_parmaters = grad_unused_zero(theta_loss, theta_model.parameters())
                t1 = time.time()
                errs = []
                for a, b in zip(vs, params):
                    diff = a - b
                    errs.append(diff)

                t2 = time.time()

                vs = [v0 - eta * (gt + gamma_1 * err) for v0, gt, err in zip(vs, grad_theta_parmaters, errs)]
                params_theta = list(theta_model.parameters())
                for param, v in zip(params_theta, vs):
                    param.update(v)
                for param in params_theta:
                    param.start_grad()
                t3 = time.time()
                # (theta, y, x, ck, gamma):
                # inner_opt = hg.GradientDescent2(low_loss_FO, step_size=args.y_lr)
                lower_loss = low_loss_FO(theta_model, y, x, ck, gamma_1)
                grad_y_parmaters = grad_unused_zero(lower_loss, list(y.parameters()))

                # last_param = inner_loop2(lower_loss, x.parameters(), y.parameters(), inner_opt, 1, log_interval=10)
                last_param = [w - args.y_lr * (g if g is not None else 0) for w, g in
                                zip(list(y.parameters()), grad_y_parmaters)]

                copy_parameter_from_list(y, last_param)


                upper_loss = upper_loss_FO(theta_model, y, x, ck)
                grad_x_parmaters = grad_unused_zero(upper_loss, list(x.parameters()))

                params_hyper = [w - args.x_lr * (g if g is not None else 0) for w, g in
                                zip(list(x.parameters()), grad_x_parmaters)]
                copy_parameter_from_list(x, params_hyper)

                t4 = time.time()

                x_time = time.time() - t0
                print(x_time)
                v = vs

                print('newmethod', x_time)

            else:
                print('NO hypergradient!')
            xgrad = [x0g for x0g in grad_x_parmaters]

            if 'BAMM' in args.hg_mode:
                time_list = [y_time, prepare_time + x_time, prepare_time + v_time]
                outer_time = max(time_list)
                step_time = outer_time
                total_hyper_time += prepare_time + x_time + v_time
                total_time += outer_time
            else:
                step_time = time.time() - t0
                total_time += time.time() - t0

            if x_itr % 1 == 0:
                xgradlist.append(copy.deepcopy([x0.detach().numpy() for x0 in xgrad]))

                with jt.no_grad():
                    # 计算并提取标量值
                    num_x = (x() - xstar) / xstar
                    print(jt.norm(num_x),)
                    x_dist = float(jt.norm(num_x).data)
                    y1_dist = float((jt.norm((y(1) - ystar)) / jt.norm(ystar)).data)
                    y2_dist = float((jt.norm((y(2) - ystar)) / jt.norm(ystar)).data)

                    print(
                        'x_itr={}, xdist={:.6f}, ydist={:.6f}, zdist={:.6f}, xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                            x_itr, x_dist, y1_dist, y2_dist,
                            xgradlist[-1][0][0],
                            total_hyper_time, total_time))

                    xlist.append(copy.deepcopy(x().detach().numpy()))
                    ylist.append(y().detach().numpy())
                    zlist.append(copy.deepcopy(y(2).detach().numpy()))
                    timelist.append(total_time)
                    alphalist.append(args.alpha)
                    print(xlist[-1])

            if jt.norm((x() - xstar) / xstar).detach().numpy() < 1e-2:
                num_x = (x() - xstar) / xstar
                print(jt.norm(num_x), )
                x_dist = float(jt.norm(num_x).data)
                y1_dist = float((jt.norm((y(1) - ystar)) / jt.norm(ystar)).data)
                y2_dist = float((jt.norm((y(2) - ystar)) / jt.norm(ystar)).data)

                print(
                    'x_itr={}, xdist={:.6f}, ydist={:.6f}, zdist={:.6f}, xgrad={:.6f}, total_hyper_time={:.6f}, total_time={:.6f}'.format(
                        x_itr, x_dist, y1_dist, y2_dist,
                        xgradlist[-1][0][0],
                        total_hyper_time, total_time))

                break
                xgradlist.append(copy.deepcopy([x0.detach().numpy() for x0 in xgrad]))

scipy.io.savemat(log_path, mdict={'x': xlist, 'y': ylist, 'z': zlist, 'time': timelist, 'n': args.tSize,
                                  'xstar': xstar.numpy(), 'F': Flist, 'xgrad': xgradlist,
                                  'eta': etalist, 'alpha': alphalist, 'Memory': jt.flags.use_cuda})
