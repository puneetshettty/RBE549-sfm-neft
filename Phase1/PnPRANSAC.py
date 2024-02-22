import numpy as np
import random
from LinearPnP import LinearPnP

def PnPRANSAC(world_points, imgj_features, K, max_iterations, threshold):
    world_points = np.array(world_points)
    imgj_features = np.array(imgj_features)
    
    best_inliers = 0
    best_R = None
    best_C = None
    best_world_points = None
    best_feature_points = None
    
    for i in range(max_iterations):
        # Ensure at least one sample can be taken
        if len(world_points) < 6:
            break
        
        indices = np.random.choice(world_points.shape[0], 6, replace=False)
        world_points_sample = world_points[indices]
        feature_points_sample = imgj_features[indices]
        R, C = LinearPnP(world_points_sample, feature_points_sample, K)
        inliers = 0
        
        for j in range(world_points.shape[0]):
            u, v,_ = imgj_features[j]
            X = np.concatenate((world_points[j], [1]))  # Add homogeneous coordinate
            # X = world_points[j].reshape(4, 1)
            C = C.reshape(3, 1)
            P = K @ R @ np.hstack((np.eye(3), -C))  # Camera projection matrix
            
            # Reproject 3D point onto image plane
            X_projected = P @ X
            u_projected = X_projected[0] / X_projected[2]
            v_projected = X_projected[1] / X_projected[2]

            # Compute reprojection error
            error = np.linalg.norm(np.array([u, v]) - np.array([u_projected, v_projected]))

            # print(f"Error: {error}")
            
            if error < threshold:
                inliers += 1
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_R = R
            best_C = C
            best_world_points = world_points_sample
            best_feature_points = feature_points_sample

    
    print(f"Best inliers: {best_inliers}")
    # Perform LinearPnP again with the best inlier features and world points
    final_R, final_C = LinearPnP(best_world_points, best_feature_points, K)

    return final_R, final_C

# def reprojectionErrorPerPoint(imgPoint,worldPoint,K,P):
#     assert worldPoint.shape == (4,1)
#     assert imgPoint.shape == (3,1)
#     p1_1t = P[0, :].reshape(1, 4)
#     p1_2t = P[1, :].reshape(1, 4)
#     p1_3t = P[2, :].reshape(1, 4)
#     u1, v1, _ = imgPoint
#     error = (u1-(np.matmul(p1_1t, worldPoint))/(np.matmul(p1_3t, worldPoint)))**2
#     + (v1-(np.matmul(p1_2t, worldPoint))/(np.matmul(p1_3t, worldPoint)))**2
#     return error


# # def PnPRANSAC(world_points, imgj_features, K, max_iterations, threshold):
# def PnPRANSAC(X_all,points,K, max_i, threshold):
#     '''
#     Input:
#         points: np.ndarray (N,3)
#         X_all: np.ndarray (N,3)
#         K: np.ndarray (3,3)
#     Output:
#         R: np.ndarray (3,3)
#         C: np.ndarray (3,1)
#         inliers: np.ndarray (N,3)
#         X: np.ndarray (N,3)
#         points: np.ndarray (N,2)
#         inliers: np.ndarray (N,1)'''
#     print(points[1])
#     print(X_all[1])
#     points = np.array(points)[:,:2]
#     X_all = np.array(X_all)[:,:3]
#     assert points.shape[1] == 2
#     assert X_all.shape[1] == 3
#     X_h = np.concatenate((X_all, np.ones((X_all.shape[0], 1))), axis=1)
#     points_h = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
#     num_iterations = 5000
#     threshold = 100
#     max_inliers = []
#     for i in range(num_iterations):
#         random_indices = random.sample(range(len(points_h)), 6)
#         points_random = points_h[random_indices]
#         X_random = X_h[random_indices]
#         # R,C = LinearPnP(points_random,X_random,K)
#         R,C = LinearPnP(X_random,points_random,K)
#         P = np.dot(K,np.dot(R,np.hstack((np.eye(3),-C))))
#         inliers = []
#         for i in range(len(points_h)):
#             point = points_h[i]
#             X= X_h[i]
#             error = reprojectionErrorPerPoint(point.reshape(3,1),X.reshape(4,1),K,P)

#             if error < threshold:
#                 inliers.append(i)
#         if len(inliers) > len(max_inliers):
#             max_inliers = inliers
#     best_kp = points[max_inliers]
#     best_X = X_all[max_inliers]
#     # R,C = LinearPnP(best_kp,best_X,K)
#     R,C = LinearPnP(best_X,best_kp, K)
#     C= C.reshape((3,1))
    
#     return R,C,best_kp,best_X, max_inliers