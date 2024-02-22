import numpy as np

def DisambiguateCameraPose(Cstack, Rstack, Xstack):

    # For all poses
    final_C = None
    final_R = None
    final_X = None
    max_inliers = []
    for C, R, X in zip(Cstack, Rstack, Xstack):
        r3 = R[:,2]
        # C = C.reshape((3,1))

        # conditions
        condition1 = X[:,2].T
        condition2 = r3 @ (X - C).T

        inliers = (condition1 > 0) & (condition2 > 0)
        # inliers = (condition2 > 0)

        if np.sum(inliers) > np.sum(max_inliers):
            final_C = C
            final_R = R
            final_X = X[inliers]
            max_inliers = inliers

    print("disambiguation")
    print(len(max_inliers))
    print(len(Xstack[0]))

    return final_C.flatten(), final_R, final_X 