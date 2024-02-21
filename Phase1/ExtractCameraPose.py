import numpy as np

def ExtractCameraPose(E):

    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    C1 = U[:,2]
    C2 = -U[:,2]
    C3 = U[:,2]
    C4 = -U[:,2]


    R1 = U @ W @ Vt
    R2 = U @ W @ Vt
    R3 = U @ W.T @ Vt
    R4 = U @ W.T @ Vt

    Cstack = [C1, C2, C3, C4]
    Rstack = [R1, R2, R3, R4]

    Cstack_fixed = []
    Rstack_fixed = []

    for Ci,Ri in zip(Cstack,Rstack):
        if np.linalg.det(Ri) < 0:
            Cstack_fixed.append(-Ci)
            Rstack_fixed.append(-Ri)
        else:
            Cstack_fixed.append(Ci)
            Rstack_fixed.append(Ri)

    return Cstack_fixed,Rstack_fixed