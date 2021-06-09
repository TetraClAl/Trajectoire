import numpy as np
from matplotlib import pyplot as plt
from rcompute import *
from display3d import *
from bras import *
from iocsvfile import *

if __name__ == "__main__":
    # Taille de la fenêtre
    tg = 45

    # Chargement d'un fichier de trajectoire
    tabdata = read_csv("D:/Weez-u/Trajectoire/data/trajectoire tube 138.csv")

    # Sélection de la fonction du bras, qui détermine sa géométrie
    f = bras2pos

    ltarget = []
    for e in tabdata:
        if e[0] != 'id' and e[0] != '-1':
            ltarget += [e[1:4]]

    # Vecteur de position de départ des actionneurs
    vec = [0, 0, 0, 0, 0, 0]

    # Calcul de la trajectoire
    S, dist = optitrajectoire2(vec, ltarget, f)

    # Graphe 3D
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

    # Graphe d'erreur
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

    # Graphe des actionneurs
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
        ax.set(ylabel='Rotation en (radians)',
               xlabel='Avancement (numéro de point)')
    for ax in axs.flat:
        ax.label_outer()

    for i in range(len(L1)):
        print(L1[i] * 180 + 180, ',', L2[i] * 180 + 180, ',', L3[i] * 180 + 180, ',',
              L4[i] * 180 + 180, ',', L5[i] * 180 + 180, ',', L6[i] * 180 + 180, ',')

    plt.show()
