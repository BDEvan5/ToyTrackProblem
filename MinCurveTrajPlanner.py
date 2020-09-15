import numpy as np
import casadi as ca 
from matplotlib import pyplot as plt 

import LibFunctions as lib 

def MinCurvatureTrajectory(track, obs_map=None):
    nvecs = track[:, 2:4]
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

    # xgrid = np.linspace(0, 100, 100)
    # ygrid = np.linspace(0, 100, 100)
    # data_flat = obs_map.ravel(order='F')
    # lut = ca.interpolant('lut', 'bspline', [xgrid, ygrid], data_flat)

    # plt.figure(1)
    # plt.imshow(obs_map, origin='lower')
    # plt.pause(0.001)

    # print(lut([45, 43]))

    n_max = 3
    N = len(track)

    n_f_a = ca.MX.sym('n_f', N)
    n_f = ca.MX.sym('n_f', N-1)
    th_f = ca.MX.sym('n_f', N-1)

    x0_f = ca.MX.sym('x0_f', N-1)
    x1_f = ca.MX.sym('x1_f', N-1)
    y0_f = ca.MX.sym('y0_f', N-1)
    y1_f = ca.MX.sym('y1_f', N-1)
    th1_f = ca.MX.sym('y1_f', N-1)
    th2_f = ca.MX.sym('y1_f', N-1)
    th1_f1 = ca.MX.sym('y1_f', N-2)
    th2_f1 = ca.MX.sym('y1_f', N-2)

    o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
    o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
    o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
    o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

    dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

    track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
                                o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

    real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
    im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

    sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan2(im(th1_f, th2_f),real(th1_f, th2_f))])
    
    get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])
    d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])

    # objective
    real1 = ca.Function('real1', [th1_f1, th2_f1], [ca.cos(th1_f1)*ca.cos(th2_f1) + ca.sin(th1_f1)*ca.sin(th2_f1)])
    im1 = ca.Function('im1', [th1_f1, th2_f1], [-ca.cos(th1_f1)*ca.sin(th2_f1) + ca.sin(th1_f1)*ca.cos(th2_f1)])

    sub_cmplx1 = ca.Function('a_cpx1', [th1_f1, th2_f1], [ca.atan2(im1(th1_f1, th2_f1),real1(th1_f1, th2_f1))])
    
    # define symbols
    n = ca.MX.sym('n', N)
    th = ca.MX.sym('th', N-1)

    nlp = {\
    'x': ca.vertcat(n, th),
    'f': ca.sumsqr(sub_cmplx1(th[1:], th[:-1])), 
    # 'f': ca.sumsqr(track_length(n)), 
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th)),
                # lut(ca.horzcat(o_x_s(n[:-1]), o_y_s(n[:-1])).T).T,

                # boundary constraints
                n[0], #th[0],
                n[-1], #th[-1],
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
        th0.append(th_00)

    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    lbx = [-n_max] * N + [-np.pi]*(N-1) 
    ubx = [n_max] * N + [np.pi]*(N-1) 

    # lbg = [0] * (N-1) + [0] * (N-1) + [0] * 2
    # ubg = [0] * (N-1) + [0.1] * (N-1) + [0] * 2

    # r = S(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    thetas = np.array(x_opt[1*N:2*(N-1)])

    # lib.plot_race_line(np.array(track), n_set, wait=True)

    return n_set

