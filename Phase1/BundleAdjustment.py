import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares



def camera_point_indices(visibility_matrix):
    """From visibility matrix, extract indices of cameras and points visible."""

    # Find non-zero indices in the visibility matrix
    non_zero_indices = np.nonzero(visibility_matrix)

    # Extract camera and point indices
    camera_indices, point_indices = non_zero_indices[1], non_zero_indices[0]

    return camera_indices, point_indices
    
def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

def project(world_points, camera_params, K):
    def projected_point(R, C, world_points, K):
        projection = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        homogenized_point = np.hstack((world_points, 1))
        x_projected = np.dot(projection, homogenized_point.T)
        x_projected /= x_projected[-1]
        return x_projected

    x_projected = []
    for i in range(len(camera_params)):
        R = Rotation.from_rotvec(camera_params[i, :3]).as_matrix()
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = world_points[i]
        pt_proj = projected_point(R, C, pt3D, K)[:2]
        x_projected.append(pt_proj)    
    return np.array(x_projected)

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    return (points_proj - points_2d).ravel()

def BundleAdjustment(visibility_matrix, world_point_cloud, feature_points, R_pnp, C_pnp, K, number_of_cameras): #IMPORTANT# add variable necessary for visibility matrix
    world_points = world_point_cloud
    points_2D = []
    visible_features = feature_points
    for point in range(visibility_matrix.shape[0]):  # point
        for camera in range(visibility_matrix.shape[1]): # camera
            if visibility_matrix[point,camera] == 1: 
                points_2D.append(visible_features[point][camera])

    points_2D = np.array(points_2D).reshape(-1,2)

    RC = []
    for i in range(number_of_cameras):
        C, R = C_pnp[i], R_pnp[i]
        Q = Rotation.from_matrix(R).as_rotvec()
        R_C = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC.append(R_C)
    RC = np.array(RC).reshape(-1,6)


    X0 = np.hstack((RC.flatten(), world_points.flatten()))
    num_world_points = world_points.shape[0]

    camera_indices, point_indices = camera_point_indices(visibility_matrix)

    A = bundle_adjustment_sparsity(number_of_cameras, num_world_points, camera_indices, point_indices)

    res = least_squares(fun, X0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(number_of_cameras, num_world_points, camera_indices, point_indices, points_2D, K))
    
    X1 = res.x
    optim_cam_param = X1[:number_of_cameras*6].reshape((number_of_cameras,6))
    optim_world_points = X1[number_of_cameras*6:].reshape((num_world_points,3))

    optim_world_map = np.zeros_like(world_point_cloud)
    optim_world_map = optim_world_points

    optim_C_set , optim_R_set = [], []
    for i in range(len(optim_cam_param)):
        R = Rotation.from_rotvec(optim_cam_param[i,:3]).as_matrix()
        C = optim_cam_param[i,3:].reshape(3,1)
        optim_C_set.append(C)
        optim_R_set.append(R)

    return optim_R_set, optim_C_set, optim_world_map
        