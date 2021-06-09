import numpy as np
from matplotlib import pyplot as plt
from rcompute import *
import random
import copy

# Caractéristiques géométriques
L1 = 40
L2 = 40
L3 = 10


# def bras1pos(ry1, rz1, rx2, ry2):
#     """ Fonction de calcul de position #1 """
#     x0 = 0
#     y0 = 0
#     l1 = L1
#     l2 = L2

#     R1 = mat_rot(0, ry1, rz1)
#     vx, vy, vz = np.dot(R1, np.array([1., 0., 0.])), np.dot(
#         R1, np.array([0., 1., 0.])), np.dot(R1, np.array([0., 0., 1.]))
#     pos1 = vx * l1

#     # print(pos1)

#     Ri = mat_rot(rx2, ry2, 0)
#     R2 = np.dot(R1, Ri)
#     # print(R2)
#     vx2, vy2, vz2 = np.dot(R2, np.array([1., 0., 0.])), np.dot(
#         R2, np.array([0., 1., 0.])), np.dot(R2, np.array([0., 0., 1.]))

#     # print(vx2)
#     pos2 = pos1 + vx2 * l2

#     return pos1, pos2
#     # print(pos2)


# def displaybras1(ry1, rz1, rx2, ry2, ax):
#     pos1, pos2 = bras1pos(ry1, rz1, rx2, ry2)

#     delta = pos2 - pos1

#     ax.quiver(0, 0, 0, pos1[0], pos1[1], pos1[2], color='b')
#     ax.quiver(pos1[0], pos1[1], pos1[2], delta[0],
#               delta[1], delta[2], color='r')


# def optidicho(f, a):
#     a2 = a + 1.2
#     a3 = a + 0.6
#     a1 = a

#     #print("a = ", a)
#     m = max(f(a1), f(a2), f(a3))
#     if m == f(a1):
#         a = a2
#         b = a3
#     else:
#         if m == f(a2):
#             b = a3
#         else:
#             b = a2
#             a += 2

#     #print("init ", [a, b])

#     while abs(a - b) > 0.000001:
#         c = (a + b)/2

#         if (f(a) - f(c)) > (f(b) - f(c)):
#             a = c
#         else:
#             b = c
#         #print("Opti : ", [a, b])

#     return (a + b) / 2


# def distbras(ry1, rz1, rx2, ry2, target):
#     pos1, pos2 = bras1pos(ry1, rz1, rx2, ry2)
#     return normv(pos2 - target)


# def optibras(ry1, rz1, rx2, ry2, target):
#     #print("Optimisation ry1")
#     def fry1(nry1): return distbras(nry1, rz1, rx2, ry2, target)
#     ry1 = optidicho(fry1, ry1)

#     #print("Optimisation rz1")
#     def frz1(nrz1): return distbras(ry1, nrz1, rx2, ry2, target)
#     rz1 = optidicho(frz1, rz1)

#     #print("Optimisation rx2")
#     def frx2(nrx2): return distbras(ry1, rz1, nrx2, ry2, target)
#     rx2 = optidicho(frx2, rx2)

#     #print("Optimisation ry2")
#     def fry2(nry2): return distbras(ry1, rz1, rx2, nry2, target)
#     ry2 = optidicho(fry2, ry2)

#     return ry1, rz1, rx2, ry2


def bras2pos(A):
    x0 = 0
    y0 = 0
    l1 = L1
    l2 = L2
    l3 = L3

    R1 = np.dot(mat_rot(0, 0, A[0]), mat_rot(0, A[1], 0))
    vx, vy, vz = np.dot(R1, np.array([1., 0., 0.])), np.dot(
        R1, np.array([0., 1., 0.])), np.dot(R1, np.array([0., 0., 1.]))
    pos1 = vx * l1

    # print(pos1)

    Ri = mat_rot(0, A[2], 0)
    R2 = np.dot(R1, Ri)
    # print(R2)
    vx2, vy2, vz2 = np.dot(R2, np.array([1., 0., 0.])), np.dot(
        R2, np.array([0., 1., 0.])), np.dot(R2, np.array([0., 0., 1.]))

    # print(vx2)
    pos2 = pos1 + vx2 * l2

    Ri = mat_rot(A[3], A[4], A[5])
    R3 = np.dot(R2, Ri)
    vx3, vy3, vz3 = np.dot(R3, np.array([1., 0., 0.])), np.dot(
        R3, np.array([0., 1., 0.])), np.dot(R3, np.array([0., 0., 1.]))

    pos3 = pos2 + vx3 * l3

    return pos1, pos2, pos3
    # print(pos2)


def bras3pos(A):
    x0 = 0
    y0 = 0
    l1 = L1
    l2 = L2
    l3 = L3

    R1 = np.dot(mat_rot(0, 0, A[0]), mat_rot(0, A[1], 0))
    vx, vy, vz = np.dot(R1, np.array([1., 0., 0.])), np.dot(
        R1, np.array([0., 1., 0.])), np.dot(R1, np.array([0., 0., 1.]))
    pos1 = vx * l1

    # print(pos1)

    Ri = mat_rot(0, A[2], 0)
    R2 = np.dot(R1, Ri)
    # print(R2)
    vx2, vy2, vz2 = np.dot(R2, np.array([1., 0., 0.])), np.dot(
        R2, np.array([0., 1., 0.])), np.dot(R2, np.array([0., 0., 1.]))

    # print(vx2)
    pos2 = pos1 + vx2 * l2

    # print(pos2)
    xy = 0
    z = 15.5
    cxy = 4.4
    cz = 1.8

    pos3 = pos2 + vz2 * np.cos(A[3]) * cxy + vy2 * \
        np.cos(A[4]) * cxy + vx2 * np.cos(A[5]) * cz + vx2 * z

    return pos1, pos2, pos3


def displaybras2(A, ax, f=bras2pos):
    pos1, pos2, pos3 = f(A)

    delta = pos2 - pos1
    beta = pos3 - pos2

    ax.quiver(0, 0, 0, pos1[0], pos1[1], pos1[2], color='b')
    ax.quiver(pos1[0], pos1[1], pos1[2], delta[0],
              delta[1], delta[2], color='r')
    ax.quiver(pos2[0], pos2[1], pos2[2], beta[0],
              beta[1], beta[2], color='g')


def distbras2(A, target, f=bras2pos):
    pos1, pos2, pos3 = f(A)
    return normv(pos3 - target)


def aleaR2(A, k):
    Ab = copy.deepcopy(A)

    for i in range(len(A)):
        Ab[i] += k * random.uniform(-1, 1)

    return Ab


# def aleaR(ry1, rz1, rx2, ry2, k):

#     ry1 += k * random.uniform(-1, 1)
#     rz1 += k * random.uniform(-1, 1)
#     rx2 += k * random.uniform(-1, 1)
#     ry2 += k * random.uniform(-1, 1)

#     return ry1, rz1, rx2, ry2


# def optibrasmonte(ry1, rz1, rx2, ry2, n, k, target):
    # d = distbras(ry1, rz1, rx2, ry2, target)

    # for i in range(n):
    #     rpy1, rpz1, rpx2, rpy2 = aleaR(ry1, rz1, rx2, ry2, k)
    #     if distbras(rpy1, rpz1, rpx2, rpy2, target) < d:
    #         d = distbras(rpy1, rpz1, rpx2, rpy2, target)
    #         ry1, rz1, rx2, ry2 = rpy1, rpz1, rpx2, rpy2

    # return ry1, rz1, rx2, ry2


def optibrasmonte2(Ain, n, k, target, f=bras2pos):
    d = distbras2(Ain, target, f)

    for i in range(n):
        Ar = aleaR2(Ain, k)
        if distbras2(Ar, target, f) < d:
            d = distbras2(Ar, target, f)
            for i in range(len(Ar)):
                Ain[i] = Ar[i]

    return Ain


def aglf(A, rot, f=bras2pos):
    pos1, pos2, pos3 = f(A)
    v1 = pos3 - pos2
    v1[0] *= rot[0]
    v1[1] *= rot[1]
    v1[2] *= rot[2]
    s = v1[0] + v1[1] + v1[2]
    return abs(np.arccos(s / (normv(rot) * normv(pos3-pos2))))


def optibrasmonte2rot(Ain, n, k, target, rot, f=bras2pos):
    d = distbras2(Ain, target, f)
    a = aglf(Ain, rot, f)

    for i in range(n):
        Ar = aleaR2(Ain, k)
        r = np.sqrt(distbras2(Ar, target, f)**2 + aglf(Ar, rot, f)**2)
        # print(r)
        if r < np.sqrt(d**2 + a**2):
            d = distbras2(Ar, target, f)
            a = aglf(Ar, rot, f)
            for i in range(len(Ar)):
                Ain[i] = Ar[i]

    return Ain


# def optimonte(ry1, rz1, rx2, ry2, target, tol=0):
#     d = distbras(ry1, rz1, rx2, ry2, target)
#     k = 2*np.pi*d/10000

#     print('OptiMonte')
#     print('k = ', k)
#     print('d = ', d)

#     e = 0

#     for i in range(20):
#         rpy1, rpz1, rpx2, rpy2 = optibrasmonte(
#             ry1, rz1, rx2, ry2, 1000, k*(0.1**e), target)
#         if (rpy1, rpz1, rpx2, rpy2) == (ry1, rz1, rx2, ry2):
#             e += 1
#             print('Reduc ', k*(0.1**e), d)
#         else:
#             ry1, rz1, rx2, ry2 = rpy1, rpz1, rpx2, rpy2
#             d = distbras(ry1, rz1, rx2, ry2, target)
#             k = 2*np.pi*d/10000
#             print('Hit   ', k*(0.1**e), d)
#         if d < tol:
#             return ry1, rz1, rx2, ry2

#     return ry1, rz1, rx2, ry2


def optimonte2(A, target, tol=0, f=bras2pos):
    d = distbras2(A, target, f)
    k = 2*np.pi*d/10000
    N = 100
    M = 20
    data = []

    print('OptiMonte')
    print('k = ', k)
    print('d = ', d)

    Ar = copy.deepcopy(A)

    for i in range(M):
        Ar2 = optibrasmonte2(Ar, N, k, target, f)

        Ar = copy.deepcopy(Ar2)

        d = distbras2(Ar, target, f)
        k = 2*np.pi*d/10000
        print('Hit   ', k, d)
        data += [[N * (i + 1), d]]
        if d < tol:
            print('Tol ', distbras2(Ar, target, f), Ar)
            return Ar

    return Ar


def optimonte2rot(A, target, rot, tol=0, f=bras2pos):
    d = distbras2(A, target, f)
    k = 2*np.pi*d/10000

    print('OptiMonte')
    print('k = ', k)
    print('d = ', d)

    Ar = copy.deepcopy(A)

    for i in range(400):
        Ar2 = optibrasmonte2rot(Ar, 400, k, target, rot, f)

        Ar = copy.deepcopy(Ar2)

        d = distbras2(Ar, target, f)
        Agl = aglf(Ar, rot, f)
        print('D', d, 'Agl', Agl)

        k = np.sqrt(Agl**2+d**2)*2*np.pi/10000
        print('Hit   ', k, d)
        if d < tol and Agl < tol:
            print('Tol ', distbras2(Ar, target, f), Ar)
            return Ar

    return Ar


# def optitrajectoire(ry1, rz1, rx2, ry2, target):
#     Lry1 = [ry1]
#     Lrz1 = [rz1]
#     Lrx2 = [rx2]
#     Lry2 = [ry2]
#     dist = []

#     target0 = [float(target[0][0]), float(target[0][1]), float(target[0][2])]
#     Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1] = optimonte(
#         Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1], target0, 0.01)

#     Lry1 += [Lry1[-1]]
#     Lrz1 += [Lrz1[-1]]
#     Lrx2 += [Lrx2[-1]]
#     Lry2 += [Lry2[-1]]
#     coord0 = [float(target[1][0]), float(target[1][1]), float(target[1][2])]
#     Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1] = optimonte(
#         Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1], coord0, 0.01)

#     for coord in target[2:]:
#         Lry1 += [2 * Lry1[-1] - Lry1[-2]]
#         Lrz1 += [2 * Lrz1[-1] - Lrz1[-2]]
#         Lrx2 += [2 * Lrx2[-1] - Lrx2[-2]]
#         Lry2 += [2 * Lry2[-1] - Lry2[-2]]

#         coord0 = [float(coord[0]), float(coord[1]), float(coord[2])]
#         print("Bras, target : ", coord0)

#         tol = 0.01
#         Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1] = optimonte(
#             Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1], coord0, tol)
#         pos1, pos2 = bras1pos(Lry1[-1], Lrz1[-1], Lrx2[-1], Lry2[-1])

#         print("Terminé : ", normv(coord0 - pos2))
#         dist += [normv(coord0 - pos2)]

#     return Lry1, Lrz1, Lrx2, Lry2, dist


def optitrajectoire2(A, target, f=bras2pos):
    """ Fonction d'optimisation position sans rotation """
    S = []
    tol = 0.0001
    vectarget = []
    dist = []

    for i in range(len(target)):
        vectarget += [[float(target[i][0]), float(target[i]
                                                  [1]), float(target[i][2])]]

    S += [optimonte2(A, vectarget[0], tol, f)]
    S += [optimonte2(S[0], vectarget[1], tol, f)]

    for i in range(2, len(vectarget)):
        B = [0, 0, 0, 0, 0, 0]
        for j in range(6):
            B[j] = 2 * S[i - 1][j] - S[i - 2][j]

        print("Bras, target : ", vectarget[i])

        B = optimonte2(B, vectarget[i], tol, f)
        S += [B]
        pos1, pos2, pos3 = bras2pos(B)

        print("Terminé : ", normv(vectarget[i] - pos3))
        dist += [normv(vectarget[i] - pos3)]

    return S, dist


def optitrajectoire2rot(A, target, rot, f=bras2pos):
    """ Fonction positon + rotation """
    S = []
    tol = 0.01
    vectarget = []
    dist = []

    for i in range(len(target)):
        vectarget += [[float(target[i][0]), float(target[i]
                                                  [1]), float(target[i][2])]]

    S += [optimonte2rot(A, vectarget[0], rot[0], tol, f)]
    S += [optimonte2rot(S[0], vectarget[1], rot[1], tol, f)]

    for i in range(2, len(vectarget)):
        B = [0, 0, 0, 0, 0, 0]
        for j in range(6):
            B[j] = 2 * S[i - 1][j] - S[i - 2][j]

        print("Bras, target : ", vectarget[i])

        B = optimonte2rot(B, vectarget[i], rot[i], tol, f)
        S += [B]
        pos1, pos2, pos3 = f(B)

        print("Terminé : ", normv(vectarget[i] - pos3))
        dist += [normv(vectarget[i] - pos3)]

    return S, dist


# --- Fonctions de tests ---

if __name__ == "__main__":

    tg = 80

    A = [0, 0, 0, 0, 0, 0]

    target = np.array([-40, -40, 0])
    A = optimonte2rot(A, target, np.array(
        [1, 0, 0]), tol=0.01, f=bras2pos)

    #A = [0.25, 0.25, 0.5, 0, 0, 0]

    pos1, pos2, pos3 = bras2pos(A)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-tg, tg])
    ax.set_ylim([-tg, tg])
    ax.set_zlim([-tg, tg])

    ax.set_xlabel('axe x')
    ax.set_ylabel('axe y')
    ax.set_zlabel('axe z')

    displaybras2(A, ax, f=bras2pos)

    plt.show()


# def test():

#     bras1pos(-0.5, 0.5, 0, 0.5)  # [1. 1. 0.]

#     tg = 80

#     target = np.array([60, 40, 0])
#     ry1, rz1, rx2, ry2 = optibras(-0.5, 0.5, 0, 0, target)
#     for i in range(1):
#         #ry1, rz1, rx2, ry2 = optibras(ry1, rz1, rx2, ry2, target)
#         #ry1, rz1, rx2, ry2 = optimonte2(ry1, rz1, rx2, ry2, target, 100000)
#         ry1, rz1, rx2, ry2 = optimonte(ry1, rz1, rx2, ry2, target)
#     pos1, pos2 = bras1pos(ry1, rz1, rx2, ry2)
#     print(ry1, rz1, rx2, ry2)
#     print(pos1, pos2)
#     print(normv(target - pos2))

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     ax.set_xlim([-tg, tg])
#     ax.set_ylim([-tg, tg])
#     ax.set_zlim([-tg, tg])

#     ax.set_xlabel('axe x')
#     ax.set_ylabel('axe y')
#     ax.set_zlabel('axe z')

#     displaybras1(ry1, rz1, rx2, ry2, ax)

#     displaybras1(0, 0, 0, 0, ax)

#     plt.show()
