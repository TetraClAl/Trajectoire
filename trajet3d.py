import numpy as np
from matplotlib import pyplot as plt
from rcompute import *
from display3d import *
from bras import *
from iocsvfile import *

if __name__ == "__main__":

    tg = 45
    tabdata = read_csv("D:/Weez-u/Trajectoire/data/trajectoire tube 138.csv")
    f = bras2pos

    ltarget = []
    for e in tabdata:
        if e[0] != 'id' and e[0] != '-1':
            ltarget += [e[1:4]]

    ltarget2 = []
    rot = []
    for i in range(20):
        ltarget2 += [[30, 50 - i * 4, 10]]
        rot += [[0, 0, 1]]

    S, dist = optitrajectoire2([0, 0, 0, 0, 0, 0], ltarget, f)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-tg, tg])
    ax.set_ylim([-tg, tg])
    ax.set_zlim([-tg, tg])

    ax.set_xlabel('axe x')
    ax.set_ylabel('axe y')
    ax.set_zlabel('axe z')

    for i in range(len(S)):
        displaybras2(S[i], ax, f)

    plt.figure(2)
    plt.plot(dist)

    plt.xlabel('Numéro de point')
    plt.ylabel('Distance à la consigne')

    L1 = []
    L2 = []
    L3 = []
    L4 = []
    L5 = []
    L6 = []
    for i in range(len(S)):
        L1 += [S[i][0]]
        L2 += [S[i][1]]
        L3 += [S[i][2]]
        L4 += [S[i][3]]
        L5 += [S[i][4]]
        L6 += [S[i][5]]

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].plot(L1)
    axs[0, 0].set_title('Moteur 1')
    axs[0, 1].plot(L2)
    axs[0, 1].set_title('Moteur 2')
    axs[0, 2].plot(L3)
    axs[0, 2].set_title('Moteur 3')
    axs[1, 0].plot(L4)
    axs[1, 0].set_title('Moteur 4')
    axs[1, 1].plot(L5)
    axs[1, 1].set_title('Moteur 5')
    axs[1, 2].plot(L6)
    axs[1, 2].set_title('Moteur 6')

    for ax in axs.flat:
        ax.set(xlabel='Rotation', ylabel='Avancement')
    for ax in axs.flat:
        ax.label_outer()

    for i in range(len(L1)):
        print(L1[i] * 180 + 180, ',', L2[i] * 180 + 180, ',', L3[i] * 180 + 180, ',',
              L4[i] * 180 + 180, ',', L5[i] * 180 + 180, ',', L6[i] * 180 + 180, ',')

    plt.show()


def test():

    bras1pos(-0.5, 0.5, 0, 0.5)  # [1. 1. 0.]

    tg = 80

    tabdata = read_csv("D:/Weez-u/Trajectoire/data/trajectoire tube 138.csv")

    ltarget = []
    for e in tabdata:
        if e[0] != 'id' and e[0] != '-1':
            ltarget += [e[1:4]]

    Lry1, Lrz1, Lrx2, Lry2, dist = optitrajectoire(-0.5, 0.5, 0, 0, ltarget)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-tg, tg])
    ax.set_ylim([-tg, tg])
    ax.set_zlim([-tg, tg])

    ax.set_xlabel('axe x')
    ax.set_ylabel('axe y')
    ax.set_zlabel('axe z')

    for i in range(len(Lry1)):
        displaybras1(Lry1[i], Lrz1[i], Lrx2[i], Lry2[i], ax)

    plt.figure(2)
    plt.plot(dist)

    plt.xlabel('Numéro de point')
    plt.ylabel('Distance à la consigne')

    plt.show()
