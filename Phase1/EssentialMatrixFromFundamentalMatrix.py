import cv2
import numpy as np
import matplotlib.pyplot as plt



def EssentialMatrixFromFundamentalMatrix(K,F):

    E = K.T.dot(F).dot(K)

    u, s, vt = np.linalg.svd(E)
    s[2] = 0
    s = np.diag(s)               
    E = np.dot(u, np.dot(s, vt))

    return E 