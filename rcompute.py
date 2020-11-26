import numpy as np


def mat_rot(rx, ry, rz):
    """ Crée une matrice de rotation rx, ry, rz. """

    rx *= np.pi
    ry *= np.pi
    rz *= np.pi

    Rmx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])

    Rmy = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

    Rmz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

    R = np.dot(Rmy, Rmz)
    return np.dot(Rmx, R)


def rep_rot(rx, ry, rz, t=1):
    """ Renvoie trois vecteurs unitaires normalisés avec une rotation rx, ry, rz. """
    v1 = np.array([t, 0, 0])
    v2 = np.array([0, t, 0])
    v3 = np.array([0, 0, t])

    R = mat_rot(rx, ry, rz)

    return np.dot(R, v1), np.dot(R, v2), np.dot(R, v3)
