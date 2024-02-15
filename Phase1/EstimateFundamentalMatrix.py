import cv2
import numpy as np
import matplotlib.pyplot as plt

def EstimateFundamentalMatrix(pts1,pts2):

    # Comverting the match points into (x1,y1) and (x2,y2) 

   
    A =[]

    for index in range(len(pts1)):

        x1 = pts1[index][0]
        y1 = pts1[index][1]
        x2 = pts2[index][0]
        y2 = pts2[index][1]

        A.append([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])


    U,S,VT = np.linalg.svd(A)

    F = VT.T[:,-1]
    
    F = F.reshape(3,3)

    # Emphasising Rank 2
    u, s, vt = np.linalg.svd(F)
    s[2] = 0
    s = np.diag(s)                    
    F = u @ s @ vt


    return F


# def main():
#     pts1 = [[36.3025, 170.408], [37.0963, 167.092], [42.2047, 200.851], [42.8895, 170.28]]
#     pts2 = [[10.2986, 150.262], [10.9623, 146.818], [19.3666, 183.084], [17.6202, 150.441]]
#     F = EstimateFundamentalMatrix(pts1,pts2)
#     print(F)

# if __name__ == "__main__":
#     data = main()