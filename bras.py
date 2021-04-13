import numpy as np
from matplotlib import pyplot as plt
from rcompute import *
import random

L1 = 40
L2 = 40


def bras1pos(ry1, rz1, rx2, ry2):
    x0 = 0
    y0 = 0
    l1 = L1
    l2 = L2

    R1 = mat_rot(0, ry1, rz1)
    vx, vy, vz = np.dot(R1, np.array([1., 0., 0.])), np.dot(
        R1, np.array([0., 1., 0.])), np.dot(R1, np.array([0., 0., 1.]))
    pos1 = vx * l1

    # print(pos1)

    Ri = mat_rot(rx2, ry2, 0)
    R2 = np.dot(R1, Ri)
    # print(R2)
    vx2, vy2, vz2 = np.dot(R2, np.array([1., 0., 0.])), np.dot(
        R2, np.array([0., 1., 0.])), np.dot(R2, np.array([0., 0., 1.]))

    # print(vx2)
    pos2 = pos1 + vx2 * l2

    return pos1, pos2
    # print(pos2)


def displaybras1(ry1, rz1, rx2, ry2, ax):
    pos1, pos2 = bras1pos(ry1, rz1, rx2, ry2)

    delta = pos2 - pos1

    ax.quiver(0, 0, 0, pos1[0], pos1[1], pos1[2], color='b')
    ax.quiver(pos1[0], pos1[1], pos1[2], delta[0],
              delta[1], delta[2], color='r')


def optidicho(f, a):
    a2 = a + 1.2
    a3 = a + 0.6
    a1 = a

    #print("a = ", a)
    m = max(f(a1), f(a2), f(a3))
    if m == f(a1):
        a = a2
        b = a3
    else:
        if m == f(a2):
            b = a3
        else:
            b = a2
            a += 2

    #print("init ", [a, b])

    while abs(a - b) > 0.000001:
        c = (a + b)/2

        if (f(a) - f(c)) > (f(b) - f(c)):
            a = c
        else:
            b = c
        #print("Opti : ", [a, b])

    return (a + b) / 2


def distbras(ry1, rz1, rx2, ry2, target):
    pos1, pos2 = bras1pos(ry1, rz1, rx2, ry2)
    return normv(pos2 - target)


def optibras(ry1, rz1, rx2, ry2, target):
    #print("Optimisation ry1")
    def fry1(nry1): return distbras(nry1, rz1, rx2, ry2, target)
    ry1 = optidicho(fry1, ry1)

    #print("Optimisation rz1")
    def frz1(nrz1): return distbras(ry1, nrz1, rx2, ry2, target)
    rz1 = optidicho(frz1, rz1)

    #print("Optimisation rx2")
    def frx2(nrx2): return distbras(ry1, rz1, nrx2, ry2, target)
    rx2 = optidicho(frx2, rx2)

    #print("Optimisation ry2")
    def fry2(nry2): return distbras(ry1, rz1, rx2, nry2, target)
    ry2 = optidicho(fry2, ry2)

    return ry1, rz1, rx2, ry2


def aleaR(ry1, rz1, rx2, ry2, k):

    ry1 += k * random.uniform(-1, 1)
    rz1 += k * random.uniform(-1, 1)
    rx2 += k * random.uniform(-1, 1)
    ry2 += k * random.uniform(-1, 1)

    return ry1, rz1, rx2, ry2


def optibrasmonte(ry1, rz1, rx2, ry2, n, k, target):
    d = distbras(ry1, rz1, rx2, ry2, target)

    for i in range(n):
        rpy1, rpz1, rpx2, rpy2 = aleaR(ry1, rz1, rx2, ry2, k)
        if distbras(rpy1, rpz1, rpx2, rpy2, target) < d:
            d = distbras(rpy1, rpz1, rpx2, rpy2, target)
            ry1, rz1, rx2, ry2 = rpy1, rpz1, rpx2, rpy2

    return ry1, rz1, rx2, ry2


def optimonte(ry1, rz1, rx2, ry2, target, tol=0):
    d = distbras(ry1, rz1, rx2, ry2, target)
    k = 2*np.pi*d/500

    print('OptiMonte')
    print('k = ', k)
    print('d = ', d)

    e = 0

    for i in range(20):
        rpy1, rpz1, rpx2, rpy2 = optibrasmonte(
            ry1, rz1, rx2, ry2, 1000, k*(0.1**e), target)
        if (rpy1, rpz1, rpx2, rpy2) == (ry1, rz1, rx2, ry2):
            e += 1
            print('Reduc ', k*(0.1**e), d)
        else:
            ry1, rz1, rx2, ry2 = rpy1, rpz1, rpx2, rpy2
            d = distbras(ry1, rz1, rx2, ry2, target)
            k = 2*np.pi*d/500
            print('Hit   ', k*(0.1**e), d)
        if d < tol:
            return ry1, rz1, rx2, ry2

    return ry1, rz1, rx2, ry2


def optimonte2(ry1, rz1, rx2, ry2, target, n):
    print('OptiMonte')

    d = distbras(ry1, rz1, rx2, ry2, target)
    k = 2*np.pi*d / 100

    print('k = ', k)
    print('d = ', d)

    for i in range(n):
        rpy1, rpz1, rpx2, rpy2 = aleaR(ry1, rz1, rx2, ry2, k)
        if distbras(rpy1, rpz1, rpx2, rpy2, target) < d:
            d = distbras(rpy1, rpz1, rpx2, rpy2, target)
            ry1, rz1, rx2, ry2 = rpy1, rpz1, rpx2, rpy2
            k = 2*np.pi*d / 100
            print('k = ', k)
            print('d = ', d)

    return ry1, rz1, rx2, ry2


def optitrajectoire(ry1, rz1, rx2, ry2, target):
    Lry1 = [ry1]
    Lrz1 = [rz1]
    Lrx2 = [rx2]
    Lry2 = [ry2]
    dist = []

    target0 = [float(target[0][0]), float(target[0][1]), float(target[0][2])]
    Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1] = optimonte(
        Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1], target0, 0.5)

    Lry1 += [Lry1[-1]]
    Lrz1 += [Lrz1[-1]]
    Lrx2 += [Lrx2[-1]]
    Lry2 += [Lry2[-1]]
    coord0 = [float(target[1][0]), float(target[1][1]), float(target[1][2])]
    Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1] = optimonte(
        Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1], coord0, 0.5)

    for coord in target[2:]:
        Lry1 += [2 * Lry1[-1] - Lry1[-2]]
        Lrz1 += [2 * Lrz1[-1] - Lrz1[-2]]
        Lrx2 += [2 * Lrx2[-1] - Lrx2[-2]]
        Lry2 += [2 * Lry2[-1] - Lry2[-2]]

        coord0 = [float(coord[0]), float(coord[1]), float(coord[2])]
        print("Bras, target : ", coord0)

        tol = 0.5
        Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1] = optimonte(
            Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1], coord0, tol)
        pos1, pos2 = bras1pos(Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1])

        print("Terminé : ", normv(coord0 - pos2))
        dist += [normv(coord0 - pos2)]

    return Lry1, Lrz1, Lrx2, Lry2, dist


if __name__ == "__main__":

    bras1pos(-0.5, 0.5, 0, 0.5)  # [1. 1. 0.]

    tg = 80

    target = np.array([60, 40, 0])
    ry1, rz1, rx2, ry2 = optibras(-0.5, 0.5, 0, 0, target)
    for i in range(1):
        #ry1, rz1, rx2, ry2 = optibras(ry1, rz1, rx2, ry2, target)
        #ry1, rz1, rx2, ry2 = optimonte2(ry1, rz1, rx2, ry2, target, 100000)
        ry1, rz1, rx2, ry2 = optimonte(ry1, rz1, rx2, ry2, target)
    pos1, pos2 = bras1pos(ry1, rz1, rx2, ry2)
    print(ry1, rz1, rx2, ry2)
    print(pos1, pos2)
    print(normv(target - pos2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-tg, tg])
    ax.set_ylim([-tg, tg])
    ax.set_zlim([-tg, tg])

    ax.set_xlabel('axe x')
    ax.set_ylabel('axe y')
    ax.set_zlabel('axe z')

    displaybras1(ry1, rz1, rx2, ry2, ax)

    plt.show()
