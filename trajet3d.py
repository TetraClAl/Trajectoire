import numpy as np
from matplotlib import pyplot as plt
from rcompute import *
from display3d import *
from bras import *
from iocsvfile import *


if __name__ == "__main__":

    bras1pos(-0.5, 0.5, 0, 0.5)  # [1. 1. 0.]

    tg = 80

    tabdata = read_csv("D:/Weez-u/Trajectoire/data/trajectoire tube 138.csv")

    ltarget = []
    for e in tabdata:
        if e[0] != 'id' and e[0] != '-1':
            ltarget += [e[1:4]]

    Lry1, Lrz1, Lrx2, Lry2 = optitrajectoire(-0.5, 0.5, 0, 0, ltarget)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-tg, tg])
    ax.set_ylim([-tg, tg])
    ax.set_zlim([-tg, tg])

    ax.set_xlabel('axe x')
    ax.set_ylabel('axe y')
    ax.set_zlabel('axe z')

    for i in range(len(Lry1)):
        displaybras1(Lry1[i], Lrz1[i], Lrx2[i], Lry2[i], ax)

    plt.show()
