import numpy as np
from matplotlib import pyplot as plt
from rcompute import *


def rep3d(ax, vc, vr, t=5):
    """ Créée dans ax un repère de centre vc et de rotation vr. """
    v1, v2, v3 = rep_rot(float(vr[0]), float(vr[1]), float(vr[2]), 5)

    #print(v1, v2, v3)

    x, y, z = float(vc[0]), float(vc[1]), float(vc[2])

    #print(x, y, z)

    ax.quiver(x, y, z, v1[0], v1[1], v1[2], color='r')
    ax.quiver(x, y, z, v2[0], v2[1], v2[2], color='g')
    ax.quiver(x, y, z, v3[0], v3[1], v3[2], color='b')


def graph3d(tabdata, tg=60):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-tg, tg])
    ax.set_ylim([-tg, tg])
    ax.set_zlim([-tg, tg])

    ax.set_xlabel('axe x')
    ax.set_ylabel('axe y')
    ax.set_zlabel('axe z')

    for e in tabdata:
        if e[0] != 'id' and e[0] != '-1':
            rep3d(ax, e[1:4], e[4:7], tg/15)

    plt.show()
