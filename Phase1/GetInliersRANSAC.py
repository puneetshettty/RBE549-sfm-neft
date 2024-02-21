import sys
import random
import numpy as np
from pprint import pprint
from matplotlib import pyplot as plt

from EstimateFundamentalMatrix import *

    
def inlier_ransac(points1, points2, indexes, max_iterations, threshold, pair):
    best_inliers = []
    points1 = np.array(points1)
    points2 = np.array(points2)
    points1 = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis=1)
    points2 = np.concatenate((points2, np.ones((points2.shape[0], 1))), axis=1)
    
    inlier_indexes = indexes

    population_size = len(points1)

    num_samples = min(8, population_size)  # Ensure num_samples is less than or equal to population_size
    # num_samples = min(7, population_size)  # Ensure num_samples is less than or equal to population_size
    best_F = np.eye(3)
    final_error = []
    is_inlier = []
    error_sum = 10000
    for _ in range(max_iterations):
        # Randomly sample a subset of points
        sample_indices = np.random.choice(population_size, size=num_samples, replace=False)
        sample_points1 = np.array([points1[i] for i in sample_indices])
        sample_points2 = np.array([points2[i] for i in sample_indices])
        
        # Estimate the fundamental matrix using the sampled points
        fundamental_matrix = EstimateFundamentalMatrix(sample_points1, sample_points2)
        
        result = points2 @ fundamental_matrix @ points1.T

        errors = np.abs(np.diag(result))
        inliers = np.where(errors < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = np.array(inliers)
            final_error = errors
            is_inlier = errors < threshold
            best_F = fundamental_matrix


        # if error_sum > np.linalg.norm(errors):
        #     print(np.linalg.norm(errors))
        #     best_inliers = np.array(inliers)
        #     final_error = errors
        #     is_inlier = errors < threshold
        #     best_F = fundamental_matrix
        #     error_sum = np.linalg.norm(errors)
 
    # Extract the inlier points from the original sets and convert them to lists
    inlier_points1 = [list(points1[i]) for i in best_inliers]
    inlier_points2 = [list(points2[i]) for i in best_inliers]

    fundamental_matrix = EstimateFundamentalMatrix(inlier_points1, inlier_points2)

    
    return best_F, inlier_points1, inlier_points2, [inlier_indexes[i] for i in best_inliers]




def get_inliers_RANSAC(matched_points):
    """
    Perform the RANSAC algorithm using the fundamental matrix to estimate inlier correspondences between image pairs
    :param matched_points: a list of the matched pixel coordinates of two images, of the form [n x 2 x 2]
    :return: the fundamental matrix with the maximum number of matched point inliers and the new list of best matches
    """

    iterations = 1000                                   # iterations of RANSAC to attempt unless found enough good paris
    epsilon = 0.05                                      # threshold for fundamental matrix transforming
    percent_good_matches = 0.99                         # what percentage of num_matches are enough to stop iterating

    matched_points = np.asarray(matched_points)         # use numpy for efficiently getting all rows of a column
    num_matches = len(matched_points)                   # number of matching feature coordinates between the images

    latest_F = np.zeros((3, 3))                         # latest computed fundamental matrix from the matched pairs

    maximum = 0                                         # how many good matches were found in the last iteration

    best_matches = []                                   # array list of best matches

    for index in range(iterations):                     # index iterator variable non-use intentional

        # pairs_indices = []

        # Select 8 matched feature pairs from each image at random
        points = [np.random.randint(0, num_matches) for num in range(8)]        # 8 random points within num_matches

        pt_1 = matched_points[points[0], 0]
        pt_2 = matched_points[points[1], 0]
        pt_3 = matched_points[points[2], 0]
        pt_4 = matched_points[points[3], 0]
        pt_5 = matched_points[points[4], 0]
        pt_6 = matched_points[points[5], 0]
        pt_7 = matched_points[points[6], 0]
        pt_8 = matched_points[points[7], 0]
        pt_p_1 = matched_points[points[0], 1]               # pt_p is an abbreviation for point_prime
        pt_p_2 = matched_points[points[1], 1]
        pt_p_3 = matched_points[points[2], 1]
        pt_p_4 = matched_points[points[3], 1]
        pt_p_5 = matched_points[points[4], 1]
        pt_p_6 = matched_points[points[5], 1]
        pt_p_7 = matched_points[points[6], 1]
        pt_p_8 = matched_points[points[7], 1]

        pts = np.array([pt_1, pt_2, pt_3, pt_4, pt_5, pt_6, pt_7, pt_8], np.float32)
        pts_prime = np.array([pt_p_1, pt_p_2, pt_p_3, pt_p_4, pt_p_5, pt_p_6, pt_p_7, pt_p_8], np.float32)

        F = EstimateFundamentalMatrix(pts, pts_prime)          # estimate the F matrix using the 8 matching paris

        num_good_matches = 0

        good_matches = []

        # Compute inliers or best matches using |x2 * F * x1| < threshold, repeat until sufficient matches found
        for i in range(num_matches):

            x_pt = matched_points[i, 0, 0]
            y_pt = matched_points[i, 0, 1]
            x_pt_prime = matched_points[i, 1, 0]
            y_pt_prime = matched_points[i, 1, 1]

            # [x, y] to [x, y, 1] for homogeneous coordinates
            point = np.array([x_pt, y_pt, 1], np.float32)
            point_prime = np.array([x_pt_prime, y_pt_prime, 1], np.float32)

            F_pt = F @ point.T
            F_pt = F_pt.T

            pt_prime_F_pt = np.multiply(point_prime, F_pt)

            error = np.sum(pt_prime_F_pt)

            if abs(error) < epsilon:
                # print(pt_prime_F_pt)
                # print('error: ', error)
                # paris_indices.append(i)
                num_good_matches += 1
                good_match = [[x_pt, y_pt], [x_pt_prime, y_pt_prime]]
                good_matches.append(good_match)

        if maximum < num_good_matches:

            maximum = num_good_matches
            print('good matches found: ', num_good_matches)

            # [matches_p.append(matched_points[ind, 0]) for ind in pairs_indices]
            # [matches_p_prime.append(matched_points[ind, 1]) for ind in pairs_indices]

            # matches_p = np.asarray(matches_p)
            # matches_p_prime = np.asarray(matches_p_prime)

            good_matches = np.asarray(good_matches)
            matches_p = good_matches[:, 0]
            matches_p_prime = good_matches[:, 1]

            # Compute the fundamental matrix for the matched pairs
            latest_F = EstimateFundamentalMatrix(matches_p, matches_p_prime)

            # Set for the output array of best matched points
            best_matches = good_matches

            if num_good_matches > percent_good_matches * num_matches:       # end if desired matches num were found
                break

    # best_matches = [matched_points[x] for x in pairs_indices]  # find the pairs corresponding to the computed indicies

    return latest_F, best_matches