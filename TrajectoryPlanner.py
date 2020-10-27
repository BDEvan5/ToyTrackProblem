import numpy as np
import casadi as ca 
from matplotlib import pyplot as plt 

import LibFunctions as lib 

def MinCurvatureTrajectory(track, obs_map=None):
    # track[:, 0:2] = track[:, 0:2] * 10
    w_min = - track[:, 4] * 0.9
    w_max = track[:, 5] * 0.9
    nvecs = track[:, 2:4]
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

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

                # boundary constraints
                n[0], #th[0],
                n[-1], #th[-1],
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
        th0.append(th_00)

    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    # lbx = [-n_max] * N + [-np.pi]*(N-1) 
    # ubx = [n_max] * N + [np.pi]*(N-1) 
    lbx = list(w_min) + [-np.pi]*(N-1) 
    ubx = list(w_max) + [np.pi]*(N-1) 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    thetas = np.array(x_opt[1*N:2*(N-1)])

    # lib.plot_race_line(np.array(track), n_set, wait=True)

    return n_set

def find_true_widths(track, check_scan_location):
    nvecs = track[:, 2:4]
    tx = track[:, 0]
    ty = track[:, 1]
    onws = track[:, 4]
    opws = track[:, 5]

    stp_sze = 0.1
    sf = 0.5 # safety factor
    N = len(track)
    nws, pws = [], []
    for i in range(N):
        pt = [tx[i], ty[i]]
        nvec = nvecs[i]

        if not check_scan_location(pt):
            j = stp_sze
            s_pt = lib.add_locations(pt, nvec, j)
            while not check_scan_location(s_pt) and j < opws[i]:
                j += stp_sze
                s_pt = lib.add_locations(pt, nvec, j)
            pws.append(j*sf)

            j = stp_sze
            s_pt = lib.sub_locations(pt, nvec, j)
            while not check_scan_location(s_pt) and j < onws[i]:
                j += stp_sze
                s_pt = lib.sub_locations(pt, nvec, j)
            nws.append(j*sf)
        else:
            print(f"Obs in way of pt: {i}")

            for j in np.linspace(0, onws[i], 10):
                p_pt = lib.add_locations(pt, nvec, j)
                n_pt = lib.sub_locations(pt, nvec, j)
                if not check_scan_location(p_pt):
                    nws.append(-j*(1+sf))
                    pws.append(opws[i])
                    break
                elif not check_scan_location(n_pt):
                    pws.append(-j*(1+sf))
                    nws.append(onws[i])
                    break 

    nws, pws = np.array(nws), np.array(pws)

    new_track = np.concatenate([track[:, 0:4], nws[:, None], pws[:, None]], axis=-1)

    return new_track



def ObsAvoidTraj(track, check_scan_location):
    track = find_true_widths(track, check_scan_location)

    w_min = - track[:, 4] * 0.9
    w_max = track[:, 5] * 0.9
    nvecs = track[:, 2:4]
    th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

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
    'f': ca.sumsqr(sub_cmplx1(th[1:], th[:-1])) * 5 + ca.sumsqr(track_length(n)), 
    # 'f': ca.sumsqr(track_length(n)), 
    'g': ca.vertcat(
                # dynamic constraints
                n[1:] - (n[:-1] + d_n(n, th)),

                # boundary constraints
                n[0], #th[0],
                n[-1], #th[-1],
            ) \
    
    }

    S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
    # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

    ones = np.ones(N)
    n0 = ones*0

    th0 = []
    for i in range(N-1):
        th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
        th0.append(th_00)

    th0 = np.array(th0)

    x0 = ca.vertcat(n0, th0)

    # lbx = [-n_max] * N + [-np.pi]*(N-1) 
    # ubx = [n_max] * N + [np.pi]*(N-1) 
    lbx = list(w_min) + [-np.pi]*(N-1) 
    ubx = list(w_max) + [np.pi]*(N-1) 

    r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

    x_opt = r['x']

    n_set = np.array(x_opt[:N])
    thetas = np.array(x_opt[1*N:2*(N-1)])

    # lib.plot_race_line(np.array(track), n_set, wait=True)

    return n_set


# def MinCurveFullState(track, obs_map=None):
#     # track[:, 0:2] = track[:, 0:2] * 10
#     w_min = - track[:, 4]
#     w_max = track[:, 5]
#     nvecs = track[:, 2:4]
#     th_ns = [lib.get_bearing([0, 0], nvecs[i, 0:2]) for i in range(len(nvecs))]

#     n_max = 3
#     N = len(track)

#     n_f_a = ca.MX.sym('n_f', N)
#     n_f = ca.MX.sym('n_f', N-1)
#     th_f = ca.MX.sym('n_f', N-1)

#     x0_f = ca.MX.sym('x0_f', N-1)
#     x1_f = ca.MX.sym('x1_f', N-1)
#     y0_f = ca.MX.sym('y0_f', N-1)
#     y1_f = ca.MX.sym('y1_f', N-1)
#     th1_f = ca.MX.sym('y1_f', N-1)
#     th2_f = ca.MX.sym('y1_f', N-1)
#     th1_f1 = ca.MX.sym('y1_f', N-2)
#     th2_f1 = ca.MX.sym('y1_f', N-2)

#     o_x_s = ca.Function('o_x', [n_f], [track[:-1, 0] + nvecs[:-1, 0] * n_f])
#     o_y_s = ca.Function('o_y', [n_f], [track[:-1, 1] + nvecs[:-1, 1] * n_f])
#     o_x_e = ca.Function('o_x', [n_f], [track[1:, 0] + nvecs[1:, 0] * n_f])
#     o_y_e = ca.Function('o_y', [n_f], [track[1:, 1] + nvecs[1:, 1] * n_f])

#     dis = ca.Function('dis', [x0_f, x1_f, y0_f, y1_f], [ca.sqrt((x1_f-x0_f)**2 + (y1_f-y0_f)**2)])

#     track_length = ca.Function('length', [n_f_a], [dis(o_x_s(n_f_a[:-1]), o_x_e(n_f_a[1:]), 
#                                 o_y_s(n_f_a[:-1]), o_y_e(n_f_a[1:]))])

#     real = ca.Function('real', [th1_f, th2_f], [ca.cos(th1_f)*ca.cos(th2_f) + ca.sin(th1_f)*ca.sin(th2_f)])
#     im = ca.Function('im', [th1_f, th2_f], [-ca.cos(th1_f)*ca.sin(th2_f) + ca.sin(th1_f)*ca.cos(th2_f)])

#     sub_cmplx = ca.Function('a_cpx', [th1_f, th2_f], [ca.atan2(im(th1_f, th2_f),real(th1_f, th2_f))])
    
#     get_th_n = ca.Function('gth', [th_f], [sub_cmplx(ca.pi*np.ones(N-1), sub_cmplx(th_f, th_ns[:-1]))])
#     d_n = ca.Function('d_n', [n_f_a, th_f], [track_length(n_f_a)/ca.tan(get_th_n(th_f))])

#     # objective
#     real1 = ca.Function('real1', [th1_f1, th2_f1], [ca.cos(th1_f1)*ca.cos(th2_f1) + ca.sin(th1_f1)*ca.sin(th2_f1)])
#     im1 = ca.Function('im1', [th1_f1, th2_f1], [-ca.cos(th1_f1)*ca.sin(th2_f1) + ca.sin(th1_f1)*ca.cos(th2_f1)])

#     sub_cmplx1 = ca.Function('a_cpx1', [th1_f1, th2_f1], [ca.atan2(im1(th1_f1, th2_f1),real1(th1_f1, th2_f1))])
    
#     # state helpers
#     d_f = ca.MX.sym('d_f', N-2)
#     v_f1 = ca.MX.sym('d_f', N-1)
#     v_f2 = ca.MX.sym('d_f', N-2)
#     dt_f = ca.MX.sym('d_f', N-2)
#     d_th = ca.Function('d_th', [d_f, v_f2], [v_f1 / 0.33 * ca.tan(d_f)])
#     d_dt = ca.Function('d_dt', [v_f1, n_f_a], [track_length(n_f_a) / v_f1])


#     # define symbols
#     n = ca.MX.sym('n', N)
#     th = ca.MX.sym('th', N-1)
#     dt = ca.MX.sym('dt', N-1)
#     v = ca.MX.sym('v', N-1)
#     # a = ca.MX.sym('a', N-1)
#     d = ca.MX.sym('d', N-1)

#     nlp = {\
#     'x': ca.vertcat(n, th, d, dt),
#     'f': ca.sumsqr(sub_cmplx1(th[1:], th[:-1])), 
#     # 'f': ca.sumsqr(track_length(n)), 
#     'g': ca.vertcat(
#                 # dynamic constraints
#                 n[1:] - (n[:-1] + d_n(n, th)),
#                 th[1:] - (th[:-1] + dt[:-1] * d_th(d[:-1], v[:-1])),
#                 dt - d_dt(v, n),

#                 # boundary constraints
#                 n[0], #th[0],
#                 n[-1], #th[-1],
#             ) \
    
#     }

#     S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':5}})
#     # S = ca.nlpsol('S', 'ipopt', nlp, {'ipopt':{'print_level':0}})

#     ones = np.ones(N)
#     n0 = ones*0

#     th0 = []
#     for i in range(N-1):
#         th_00 = lib.get_bearing(track[i, 0:2], track[i+1, 0:2])
#         th0.append(th_00)

#     th0 = np.array(th0)

#     dt0 = track_length(n0) / 

#     x0 = ca.vertcat(n0, th0, d0, dt0)

#     # lbx = [-n_max] * N + [-np.pi]*(N-1) 
#     # ubx = [n_max] * N + [np.pi]*(N-1) 
#     N1 = N-1
#     lbx = list(w_min) + [-np.pi]*(N-1) + [-0.4]*N1 + [0]*N1
#     ubx = list(w_max) + [np.pi]*(N-1) + [0.4]*N1 + [0.5]*N1

#     r = S(x0=x0, lbg=0, ubg=0, lbx=lbx, ubx=ubx)

#     x_opt = r['x']

#     n_set = np.array(x_opt[:N])
#     thetas = np.array(x_opt[1*N:2*(N-1)])

#     lib.plot_race_line(np.array(track), n_set, wait=True)

#     return n_set


def generate_velocities(wpts):
    N = len(wpts)

    bearings = [lib.get_bearing(wpts[i], wpts[i+1]) for i in range(N-1)]
    dths = [abs(lib.sub_angles_complex(bearings[i+1], bearings[i])) for i in range(N-2)]
    dths.insert(0, 0.001)
    dths.append(0.001)
    dxs = [lib.get_distance(wpts[i], wpts[i+1]) for i in range(N-1)]

    max_v = 7.5
    max_a = 7.5
    min_a = -8.5
    mass = 3.74
    l = 0.33
    mu = 0.523
    mu_sf = 0.9*mu

    f_max_steer = mu_sf * mass

    # plt.plot(dths)
    # plt.show()

    max_vs = [min(f_max_steer/(mass*dths[i]), max_v) for i in range(N)]

    v_outs = [max_vs[-1]]
    for i in reversed(range(N-1)):
        v_min = v_outs[-1] - 2 * max_a * dxs[i]
        v_max = v_outs[-1] - 2 * min_a * dxs[i]
        v = max_vs[i+1]
        v = np.clip(v, v_min, v_max)
        v_outs.append(v)

    v_outs = np.array(v_outs)

    return v_outs
    # max_vs = np.array(max_vs)
    # print(max_vs)
    # plt.figure(1)
    # plt.plot(max_vs)
    # plt.pause(0.001)
    # plt.figure(2)
    # plt.plot(v_outs)
    # plt.pause(0.001)
    # plt.figure(3)
    # plt.plot(v_outs[1:] - v_outs[:-1])
    # plt.plot(max_vs[1:] - max_vs[:-1])
    # plt.show()



