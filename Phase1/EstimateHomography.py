import numpy as np
import heapq
import random

def calculate_homography(src_points, dest_points):
    # Shape: (n, 2)
    src_points = np.array(src_points)
    dest_points = np.array(dest_points)

    # Construct the A matrix for Ax=0
    A = []
    for i in range(len(src_points)):
        x, y = src_points[i]
        u, v = dest_points[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])

    A = np.array(A)

    # Perform Singular Value Decomposition (SVD)
    _, _, V = np.linalg.svd(A)

    # Homography matrix is the last column of V
    H = V[-1, :].reshape(3, 3)
    H = H * 1 / H[2, 2]

    return H


def get_homography_ransac(kp1, kp2, matches, tau = 40, n_iter = 10000,
           lower_bound_of_reduction = 20,
           number_of_matching_quads = 6,
           tau_upper_bound = 90,
           tau_increments = 10,
           tau_increment_period = 10000,
           last_ditch_effort = 40,
           ):

    heuristic_cutoff_iter = np.array([10000, 20000, 30000, 40000, 50000])
    heuristic_cutoff_score = np.array([1000, 400, 300, 200, 100])


    data_points = []
    i = 0
    run_till_found = True
    if len(matches) < 4:
        print('\t ⨯ Not enough matches')
        return None, None, None, 0

    if len(matches) < lower_bound_of_reduction:
        print('\t ⨯ Reduce n_iter due to less matches')
        n_iter = 400 * len(matches)
    while run_till_found:
        i += 1
        selected_matches = random.choices(matches, k=4)

        src = np.array([i for (i,j) in selected_matches],np.float32)
        dst = np.array([j for (i,j) in selected_matches],np.float32)

        try:
            H = calculate_homography(src, dst)
        except:
            continue
        source_points = np.array([[x, y, 1] for x,y in src]).T
        dest_points = np.array([[x, y, 1] for x,y in dst]).T

        src_dash = H @ source_points

        delta = np.sum(np.square(dest_points - src_dash))
        if delta > 1000 and i < n_iter:
            continue

        f1 = False
        while not f1:
            try:
                heapq.heappush(data_points, (delta, src, dst))
                f1 = True
            except:
                delta += random.random()

        top_n_points = heapq.nsmallest(number_of_matching_quads, data_points, key=lambda x: x[0])
        # if len(top_n_points) >= 4 and top_n_points[3][0] < 16:
        #     top_n_points = [x for x in heapq.nsmallest(min(10, len(data_points)), data_points, key=lambda x: x[0]) if x[0] < 90]
        #     break

        if i >= n_iter:
            if top_n_points[0][0] <= last_ditch_effort:
                print('\t - Selecting Smallest Delta')
                top_n_points = [x for x in heapq.nsmallest(min(10, len(data_points)), data_points, key=lambda x: x[0]) if
                                x[0] < 90]
                break

            else:
                print('\t ⨯ Reached max iterations')
                return None, None, None, 0

        if top_n_points[-1][0] < tau:
            run_till_found = False

        if i % tau_increment_period == 0:
            if tau < tau_upper_bound:
                tau += tau_increments

        # Unlikely to match at this point
        if np.any(np.logical_and(i > heuristic_cutoff_iter, top_n_points[0][0] > heuristic_cutoff_score)):
            print('\t ⨯ Heuristically unlikely to match')
            return None, None, None, 0


    inlier_src = [x[1] for x in top_n_points]
    inlier_dst = [x[2] for x in top_n_points]

    # Reshape so that its (n,2) instead of (7,4,2)
    inlier_src = (np.reshape(inlier_src, (-1,2)))
    inlier_dst = (np.reshape(inlier_dst, (-1,2)))

    confidence_array = [x[0] for x in top_n_points]
    confidence = sum(confidence_array) *1.0 / len(confidence_array)
    return calculate_homography(inlier_src, inlier_dst), inlier_src, inlier_dst, round(confidence.item(),2)