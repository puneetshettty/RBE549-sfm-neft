import numpy as np
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


