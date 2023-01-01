import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit

A1 = 150000
E1 = 78000

A2 = 5 * 10 ** 6
E2 = 87900

R = 8.31

u = 0.01
ro = 2.5

C1_vh = 0.7
C2_vh = 0.3


# def C_change(C, mol_m)
#   Cvh = C * ro / 100% * mol_m
@jit
def math_model(Lt, TT):
    result_1 = list()
    result_2 = list()
    result_3 = list()

    L = list()
    d_t = 0.01
    result_1.append(70)
    result_2.append(30)
    result_3.append(0.0)
    cnt = -1

    L.append(Lt)

    k1 = lambda TT: A1 * np.exp((-E1) / (R * TT))
    k2 = lambda TT: A2 * np.exp((-E2) / (R * TT))

    d_C3 = lambda C1,C2, C3, t, u: (-k2(t) * C1 * C3 + k1(t) * C1 * C2) / u
    d_C1 = lambda C1, C2, C3, t, u: (-k1(t) * C1 * C2 - k2(t) * C1 * C3) / u
    d_C2 = lambda C1, C2, t, u: (-k1(t) * C1 * C2) / u

    while True:
        cnt += 1
        result_3.append(result_3[-1] + d_C3(result_1[-1], result_2[-1], result_3[-1], TT, u) * d_t)
                # print(result_3)
        result_1.append(result_1[-1] + d_C1(result_1[-1], result_2[-1], result_3[-1], TT, u) * d_t)
        result_2.append(result_2[-1] + d_C2(result_1[-2], result_2[-1], TT, u) * d_t)
        L.append(L[-1] + d_t)

        return L[-1], TT, result_3[-1]


def minimum():
    L = 0
    d_t = 0.3
    TT_range = np.linspace(440 , 480, 30)
    L_list = list()
    TT_list = list()
    C_list = list()
    for i in TT_range:
        tempL, tempTT, tempC = math_model(L, i)
        L += d_t
        L_list.append(tempL)
        TT_list.append(tempTT)
        C_list.append(tempC)

    print(C_list)
    print(TT_list)
    print(L_list)
    #max_elem = C_list[0][0]
    #n, m = 0, 0
    #for i in range(len(C_list)):
    #    for j in range(len(C_list[i])):
        #       if C_list[i][j] > max_elem:
        #          max_elem = C_list[i][j]
    #           n, m = i, j
    #print(max_elem)
    #print(L_list[n][m])
    #print(TT_list[n])
    #c_max = max(C_list[])
    #min_l = L_list[][C_list[].index(c_max)]
    #print(min_l)
    #plot(L_list[n], L_list[n], C_list[n])
    #plot3D(L_list, TT_list, C_list)
    # return minTime, minF
def plot(Time,y,C):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), label='Coursework')
    axes[0].plot(Time, y, color='black')
    axes[0].grid(True)
    axes[0].set(xlabel="Время(с)", ylabel="")
    axes[0].axis(ymin=7,ymax=20)
    axes[1].plot(Time, C, color='black')
    axes[1].grid(True)
    axes[1].set(xlabel="Время(с)", ylabel="Концентрация(моль/м^3)")
    # axes[1].axis(ymin=0.1,ymax=0.31)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.show()

# def imitation(Tt):
#     time, numbers = lab_6()
#     numbers.insert(0, Tt)
#     minTime, minF = minimum()
#     print(f'Минимальное время: {minTime}')
#     print(f'Минимальная площадь: {minF}')
#     List_C = list()
#     for i in numbers:
#         tempTime, tempF, tempT, tempC = math_model(minF, i + 273)
#         List_C.append(tempC)
#
#     plot(time, numbers[:990], List_C[:990])


# def lab_6():
#     lam1 = 5 ** 12
#     lam2 = 3 ** 11
#     M0 = 15
#     disp0 = 4
#     alpha0 = -0.085
#     N = 1000
#     Ns = 10
#     x = [1]
#     xi = []
#     for j in range(N):
#         x.append(((lam1 * x[j]) % lam2))
#         xi.append(x[j] / lam2 - 0.5)
#     Mx = 1 / N * sum(xi)
#     peremen = 0
#     for j in range(N):
#         peremen += (xi[j] - Mx) ** 2
#     dispx = 1 / N * peremen
#     z = []
#     for k in range(1, N - 9):
#         j = 0
#         for i in np.arange(k, k + Ns):
#             j += xi[i] * np.sqrt(disp0 / (dispx * -alpha0)) * np.exp(alpha0 * (i - k)) + M0
#         z.append(1 / Ns * j)
#     x1 = [i for i in range(N)]
#     z1 = [i for i in range(N - 10)]
#     return z1, z


# def plot(Time, y, C):
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), label='Coursework')
#     axes[0].plot(Time, y, color='black')
#     axes[0].grid(True)
#     axes[0].set(xlabel="Время(с)", ylabel="")
#     axes[0].axis(ymin=7, ymax=20)
#     axes[1].plot(Time, C, color='black')
#     axes[1].grid(True)
#     axes[1].set(xlabel="Время(с)", ylabel="Концентрация(моль/м^3)")
#     axes[1].axis(ymin=0.1, ymax=0.31)
#     plt.subplots_adjust(wspace=0.5, hspace=0)
#     plt.show()


def plot3D(F, t, C):
    fig = plt.figure(figsize=(10, 6))
    axes = Axes3D(fig, auto_add_to_figure=False)
    axes.plot_trisurf(F, t, C, cmap='plasma', edgecolor='none', antialiased=True)
    axes.set_xlabel('F')
    axes.set_ylabel('τ')
    axes.set_zlabel('C')
    fig.add_axes(axes)
    plt.show()


# def main():
#     imitation(8)


if __name__ == '__main__':
    minimum()
