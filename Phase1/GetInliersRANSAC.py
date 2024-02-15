import random
import numpy as np

from EstimateFundamentalMatrix import *

# def inlier_ransac(points1, points2, max_iterations, threshold):
#     best_inliers = []
#     points1 = np.concatenate((points1,np.ones((points1.shape[0],1))),axis=1)
#     points2 = np.concatenate((points2,np.ones((points2.shape[0],1))),axis=1)
    
def inlier_ransac(points1, points2, indexes, max_iterations, threshold):
    best_inliers = []
    points1 = np.array(points1)
    points2 = np.array(points2)
    points1 = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis=1)
    points2 = np.concatenate((points2, np.ones((points2.shape[0], 1))), axis=1)
    
    inlier_indexes = indexes

    population_size = len(points1)
    num_samples = min(8, population_size)  # Ensure num_samples is less than or equal to population_size
    
    for _ in range(max_iterations):
        # Randomly sample a subset of points
        sample_indices = np.random.choice(population_size, size=num_samples, replace=False)
        sample_points1 = np.array([points1[i] for i in sample_indices])
        sample_points2 = np.array([points2[i] for i in sample_indices])
        
        # Estimate the fundamental matrix using the sampled points
        fundamental_matrix = EstimateFundamentalMatrix(sample_points1, sample_points2)
        
        # Compute the error for all points
        errors = np.abs(np.sum(points2 * (fundamental_matrix @ np.array(points1).T).T, axis=1))
        
        # Check which points are inliers (i.e., have error below the threshold)
        inliers = np.where(errors < threshold)[0]
        
        # Update best set of inliers if the current set is better
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            
    # Extract the inlier points from the original sets and convert them to lists
    inlier_points1 = [list(points1[i]) for i in best_inliers]
    inlier_points2 = [list(points2[i]) for i in best_inliers]
    
    return fundamental_matrix, inlier_points1, inlier_points2, [inlier_indexes[i] for i in best_inliers]



# def main():
#     pts1 = [[36.3025, 170.408], [37.0963, 167.092], [42.2047, 200.851], [42.8895, 170.28], [43.4867, 191.131], [47.8875, 196.202], [49.336, 203.336], [49.6084, 542.71], [56.786, 202.413]]
#     pts2 = [[10.2986, 150.262], [10.9623, 146.818], [19.3666, 183.084], [17.6202, 150.441], [19.588, 172.699], [24.261, 178.192], [26.2842, 185.986], [220.36, 515.917], [34.2992, 184.947]]
#     inlier_points1, inlier_points2 = inlier_ransac(pts1, pts2, 1000, 0.5)
#     print(inlier_points1)
#     print(inlier_points2)

# if __name__ == "__main__":
#     main()