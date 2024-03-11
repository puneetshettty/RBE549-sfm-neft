import glob
import json
import sys, os
import math
import random

import imageio.v3 as iio

import imageio
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

from log_setup import logger
from NeRFModel import *
import argparse

# model = NeRFmodel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
np.random.seed(0)

def meshgrid_xy(tensor1, tensor2):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def cumprod_exclusive(tensor):
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
        tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.
    
    Returns:
        cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.
    
    return cumprod

def get_ray_bundle(height, width, focal_length, tform_cam2world):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

    Args:
        height (int): Height of an image (number of pixels).
        width (int): Width of an image (number of pixels).
        focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
        tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera frame to the "world" frame for the current example.
    
    Returns:
        ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
        each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
        row index `j` and column index `i`.
        (TODO: double check if explanation of row and col indices convention is right).
        ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
        direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
        passing through the pixel at row index `j` and column index `i`.
        (TODO: double check if explanation of row and col indices convention is right).
    """
    # TESTED
    ii, jj = meshgrid_xy(
        torch.arange(width).to(tform_cam2world),
        torch.arange(height).to(tform_cam2world)
    )
    directions = torch.stack([(ii - width * .5) / focal_length,
                                -(jj - height * .5) / focal_length,
                                -torch.ones_like(ii)
                            ], dim=-1)
    ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
    return ray_origins, ray_directions

def compute_query_points_from_rays(
    ray_origins,
    ray_directions,
    near_thresh,
    far_thresh,
    num_samples,
    randomize
):
  r"""Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
  variables indicate the bounds within which 3D points are to be sampled.

  Args:
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
      coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
      coordinate that is of interest/relevance).
    num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
      randomly, whilst trying to ensure "some form of" uniform spacing among them.
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
      By default, this is set to `True`. If disabled (by setting to `False`), we sample
      uniformly spaced points along each ray in the "bundle".
  
  Returns:
    query_points (torch.Tensor): Query points along each ray
      (shape: :math:`(width, height, num_samples, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).
  """
  # TESTED
  # shape: (num_samples)
  depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
  if randomize is True:
    # ray_origins: (width, height, 3)
    # noise_shape = (width, height, num_samples)
    noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
    # depth_values: (num_samples)
    depth_values = depth_values \
        + torch.rand(noise_shape).to(ray_origins) * (far_thresh
            - near_thresh) / num_samples
  # (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
  # query_points:  (width, height, num_samples, 3)
  query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
  # TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
  return query_points, depth_values

def render_volume_density(
    radiance_field,
    ray_origins,
    depth_values
):
  r"""Differentiably renders a radiance field, given the origin of each ray in the
  "bundle", and the sampled depth values along them.

  Args:
    radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
      we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
      the paper) (shape: :math:`(width, height, num_samples, 4)`).
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).
  
  Returns:
    rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
    depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
    acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
      transmittance map).
  """
  # TESTED
  sigma_a = torch.nn.functional.relu(radiance_field[..., 3])
  rgb = torch.sigmoid(radiance_field[..., :3])
  one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
  dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],
                  one_e_10.expand(depth_values[..., :1].shape)), dim=-1)
  alpha = 1. - torch.exp(-sigma_a * dists)
  weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

  rgb_map = (weights[..., None] * rgb).sum(dim=-2)
  depth_map = (weights * depth_values).sum(dim=-1)
  acc_map = weights.sum(-1)

  return rgb_map, depth_map, acc_map


def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True):
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

def get_minibatches(inputs, chunksize):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

# One iteration of TinyNeRF (forward pass).
def run_one_iter_of_tinynerf(height, width, focal_length, tform_cam2world,
                             near_thresh, far_thresh, depth_samples_per_ray,
                             encoding_function, get_minibatches_function):
  
    # Get the "bundle" of rays through all image pixels.
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length,
                                                tform_cam2world)
    
    # Sample query points along each ray
    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    # "Flatten" the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points (default: positional encoding).
    encoded_points = encoding_function(flattened_query_points)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, _, _ = render_volume_density(radiance_field, ray_origins, depth_values)

    return rgb_predicted

def loadDataset(data_path, mode):
    """
    Input:
        data_path: dataset path
        mode: train or test
    Outputs:
        camera_info: image width, height, camera matrix 
        images: images
        pose: corresponding camera pose in world frame
    """
    # check if file exists
    transforms_file = os.path.join(data_path, f"transforms_{mode}.json")
    if not os.path.exists(transforms_file):
        logger.error("Dataset not found")
        raise FileNotFoundError(f"Dataset {mode} not found at {transforms_file}")

    json_data = json.load(open(transforms_file))

    # TODO need to move these data blocks to GPU using torch .to()
    camera_angle_x = json_data['camera_angle_x']

    # TODO need to check if this is correct
    focal = 0.5 / np.tan(0.5 * camera_angle_x)
    print(focal, camera_angle_x)
    sys.exit()

    frames = json_data['frames']
    images = [iio.imread(os.path.join(data_path, frame['file_path']+'.png')) for frame in frames]
    poses = np.array([np.array(frame['transform_matrix']) for frame in frames])

    height, width = images[0].shape[:2]
    # TODO need to check if this is correct
    camera_matrix = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]])

    return (width, height, camera_matrix), images, poses

    
def PixelToRay(W, H, focal, pose):
    """
    Input:
        camera_info: image width, height, camera matrix 
        pose: camera pose in world frame
        pixelPoition: pixel position in the image
        args: get near and far range, sample rate ...
    Outputs:
        ray origin and direction
    """
    x = torch.arange(W).to(device)
    y = torch.arange(H).to(device)
    x, y = torch.meshgrid(x, y)
    x = x.transpose(-1, -2)
    y = y.transpose(-1, -2) 


    x = (x - H * 0.5) / focal
    y = (y - W * 0.5) / focal

    ray_direction = torch.stack([x, -y, -torch.ones_like(x)], -1)

    Rotation = pose[:3, :3]
    Translation = pose[:3, -1]
    # Rotation = torch.tensor(Rotation_orig, device=device)
    # Translation = torch.tensor(Translation_orig, device=device)

    directions = ray_direction[..., None, :]
    ray_directions = torch.sum(directions * Rotation, -1)
    ray_origins = Translation.expand(ray_directions.shape)
    # ray_origins = torch.broadcast_to(Translation, ray_directions.shape)

    return ray_origins, ray_directions

def SamplePoints(ray_origins, ray_directions, near, far, N_samples):
    """
    Input:
        ray_origins: origins of input rays
        ray_directions: direction of input rays
        near: near range
        far: far range
        N_samples: number of sample per ray
    Outputs:
        sampled points
    """
    num_rays = ray_origins.shape[0]
    samples = torch.linspace(near, far, N_samples, device=device)
    # z_vals = 1.0 / (1.0 / near * (1.0 - samples) + 1.0 / far * samples)
    # z_vals = z_vals.expand([num_rays, N_samples])
    rays = ray_origins[..., None, :] + samples[..., None] * ray_directions[..., None, :]
    # print("Rays shape: ", rays.shape)
    points = rays.to(device)
    

    return points, samples


def generateBatch(raypoints, batch_size):
    """
    Input:
        raypoints: sampled points
        batch_size: number of rays per batch
    Outputs:
        batched points
    """
    batches = []  # Define the "inputs" variable as an empty list
    for i in range(0, len(raypoints), batch_size):
        batches.extend(raypoints[i: i + batch_size])  # Append the batched points to the "inputs" list

    return batches
     

# To calculate the cumulative product used to calculate alpha
def cumprod_exclusive(tensor) :
    dim = -1
    cumprod = torch.cumprod(tensor, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0] = 1.
    
    return cumprod

def render(radiance_field, ray_directions, depth_values):
    """
    Input:
        radiance_field: rgb from model
        ray_directions: direction of input rays
        depth_values: number of samples
    Outputs:
        rgb_map: rgb color of n depth values
        depth_map: depth map
        acc_map: accumulated map
    """
    sigma_a = F.relu(radiance_field[...,3])       #volume density
    rgb = torch.sigmoid(radiance_field[...,:3])    #color value at nth depth value
    one_e_10 = torch.tensor([1e10], dtype = ray_directions.dtype, device = ray_directions.device)
    # print("one_e_10", one_e_10.shape)
    dists = torch.cat((depth_values[...,1:] - depth_values[...,:-1], one_e_10.expand(depth_values[...,:1].shape)), dim = -1)
    alpha = 1. - torch.exp(-sigma_a * dists)       
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)     #transmittance
    rgb_map = (weights[..., None] * rgb).sum(dim = -2)          #resultant rgb color of n depth values
    depth_map = (weights * depth_values).sum(dim = -1)
    acc_map = weights.sum(-1)
    print("rgb_map", rgb_map.shape)

    return rgb_map, depth_map, acc_map


def loss_function(groundtruth, prediction):
    loss = torch.mean((groundtruth - prediction)**2)
    return loss


def load_latest_model(chk_path, model, optimizer):
    list_of_files = glob.glob(os.path.join(chk_path, '*'))
    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        checkpoint = torch.load(latest_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def render_image(model, poses, camera_info, args, i):
    model.eval()
    with torch.no_grad():
        pose = torch.tensor(poses[i], device=device)
        camera_inform = camera_info
        

        # Get the ray origins and directions
        ray_origins, ray_directions = PixelToRay(camera_inform[0], camera_inform[1], camera_inform[2][0,0], pose)
        ray_origins = ray_origins.to(device)
        ray_directions = ray_directions.to(device)

        x = torch.arange(camera_inform[0], device=pose.device)
        y = torch.arange(camera_inform[1], device=pose.device)
        x, y = torch.meshgrid(x, y)
        x = x.transpose(-1, -2)
        y = y.transpose(-1, -2) 
        coords = torch.stack((x, y), dim=-1)
        coords = coords.reshape((-1, 2))
        ray_idx = np.random.choice(coords.shape[0], size = args.n_rays, replace=False)
        ray_idx = coords[ray_idx]
        ray_origin = ray_origins[ray_idx[:,0], ray_idx[:,1], :]
        ray_direction = ray_directions[ray_idx[:,0], ray_idx[:,1], :]

        viewdirs = ray_direction/ ray_direction.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view(-1, 3)

        ray_origin = ray_origin.view(-1, 3)
        ray_direction = ray_direction.view(-1, 3)

        # near = 2.0 * torch.ones_like(ray_origin[..., :1])
        # far = 6.0 * torch.ones_like(ray_origin[..., :1])

        # Sample the points
        points, samples = SamplePoints(ray_origin, ray_direction, 2, 6, args.n_sample)
        # print(f'Points shape: {points.shape}, Viewdirs shape: {viewdirs.shape}')


        input_dirs = viewdirs.unsqueeze(1).expand(points.shape)
        input_dirs = input_dirs.reshape(-1, 3)
        points_flat = points.reshape(-1, 3)
        embedded_pts = positional_encoding(points_flat, num_encoding_functions=6, include_input=True, log_sampling=True)
        embedded_dirs = positional_encoding(input_dirs, num_encoding_functions=args.n_dirc_freq, include_input=True, log_sampling=True)
        # print(f'Embedded points shape: {embedded_pts.shape}, Embedded dirs shape: {embedded_dirs.shape}')

        embed = torch.cat((embedded_pts, embedded_dirs), -1)
        # print(f'Embed shape: {embed.shape}')

        # Generate the batch
        # print("Batch being generated")
        batches = generateBatch(embed, args.batchsize)
        # print("Batch generated")
        # print(f'number of batches: {len(batches)}')


        # Forward pass
        prediction = [model(batch) for batch in batches]
        # print(f'shape of prediction: {prediction[0].shape}')

        radiance_field = torch.stack(prediction, dim=0)
        # print(f'Radiance field shape: {radiance_field.shape}')
        radiance_field = radiance_field.reshape(list(points.shape[:-1]) + [radiance_field.shape[-1]])

        # Render the image
        rgb_map, depth_map, acc_map = render(radiance_field, ray_direction, samples)

    return rgb_map, ray_idx


def validate(model, image, poses, camera_info, args, idx):
    rgb_map, ray_idx = render_image(model, poses, camera_info, args, idx)
    # Calculate the loss
    target_img = image[ray_idx[:,0], ray_idx[:,1], :]
    return loss_function(rgb_map[..., :3], target_img[..., :3])


def train(images, poses, camera_info, args):
    """
    Input:
        images: images
        poses: camera pose in world frame
        camera_info: image width, height, camera matrix 
        args: training arguments
    """
    # Initialize the model
    model = NeRFmodel()
    model.to(device)
    model.train()

    # Initialize the optimize3r
    optimizer = torch.optim.Adam(model.parameters(), args.lrate)

    # Initialize the summary writer
    writer = SummaryWriter(args.logs_path)

    # Load the checkpoint if required
    if args.load_checkpoint:
        load_latest_model(args.checkpoint_path, model, optimizer)

    # Initialize the loss
    loss = 0

    # Start the training loop
    for i in range(args.max_iters):
        # Sample a random image
        idx = random.randint(0, len(images)-1)
        image = torch.tensor(images[idx], device=device)
        pose = torch.tensor(poses[idx], device=device)
        camera_inform = camera_info
    

        # Get the ray origins and directions
        ray_origins, ray_directions = PixelToRay(camera_inform[0], camera_inform[1], camera_inform[2][0,0], pose)
        ray_origins = ray_origins.to(device)
        ray_directions = ray_directions.to(device)

        x = torch.arange(camera_inform[0], device=pose.device)
        y = torch.arange(camera_inform[1], device=pose.device)
        x, y = torch.meshgrid(x, y)
        x = x.transpose(-1, -2)
        y = y.transpose(-1, -2) 
        coords = torch.stack((x, y), dim=-1)
        coords = coords.reshape((-1, 2))
        ray_idx = np.random.choice(coords.shape[0], size = args.n_rays, replace=False)
        ray_idx = coords[ray_idx]
        ray_origin = ray_origins[ray_idx[:,0], ray_idx[:,1], :]
        ray_direction = ray_directions[ray_idx[:,0], ray_idx[:,1], :]

        target_img = image[ray_idx[:,0], ray_idx[:,1], :]

        viewdirs = ray_direction/ ray_direction.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view(-1, 3)

        ray_origin = ray_origin.view(-1, 3)
        ray_direction = ray_direction.view(-1, 3)

        # near = 2.0 * torch.ones_like(ray_origin[..., :1])
        # far = 6.0 * torch.ones_like(ray_origin[..., :1])

        # Sample the points
        points, samples = SamplePoints(ray_origin, ray_direction, 2, 6, args.n_sample)
        # print(f'Points shape: {points.shape}, Viewdirs shape: {viewdirs.shape}')


        

        input_dirs = viewdirs.unsqueeze(1).expand(points.shape)
        # print(f'Input dirs shape: {input_dirs.shape}')
        input_dirs = input_dirs.reshape(-1, 3)
        points_flat = points.reshape(-1, 3)
        embedded_pts = positional_encoding(points_flat, num_encoding_functions=6, include_input=True, log_sampling=True)
        embedded_dirs = positional_encoding(input_dirs, num_encoding_functions=args.n_dirc_freq, include_input=True, log_sampling=True)
        # print(f'Embedded points shape: {embedded_pts.shape}, Embedded dirs shape: {embedded_dirs.shape}')

        embed = torch.cat((embedded_pts, embedded_dirs), -1)
        # print(f'Embed shape: {embed.shape}')

        # Generate the batch
        # print("Batch being generated")
        batches = generateBatch(embed, args.batchsize)
        # print("Batch generated")
        # print(f'number of batches: {len(batches)}')




        # Forward pass
        prediction = [model(batch) for batch in batches]
        # print(f'shape of prediction: {prediction[0].shape}')

        radiance_field = torch.stack(prediction, dim=0)
        # print(f'Radiance field shape: {radiance_field.shape}')
        radiance_field = radiance_field.reshape(list(points.shape[:-1]) + [radiance_field.shape[-1]])

        # Render the image
        rgb_map, depth_map, acc_map = render(radiance_field, ray_direction, samples)

        # Calculate the loss
        loss = loss_function(rgb_map[..., :3], target_img[..., :3]).to(device)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log the loss
        writer.add_scalar('Loss', loss, i)

        # Save the checkpoint
        if i % args.save_ckpt_iter == 0:
            checkpoint_save_name =  args.checkpoint_path + os.sep + 'model_' + str(i) + '.ckpt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_save_name)
        
        # Validate the model
        loss = validate(model, image, poses, camera_info, args, idx)
        writer.add_scalar('val_Loss', loss, i)

        print(f'Iteration: {i}, Loss: {loss}')

    # Close the summary writer
    writer.close()

    return model

    
def test(images, poses, camera_info, args):
    # Initialize the model
    model = NeRFmodel()
    model.to(device)
    model.train()

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lrate)

    # Initialize the summary writer
    writer = SummaryWriter(args.logs_path)

    # Load the checkpoint if required
    if args.load_checkpoint:
        load_latest_model(args.checkpoint_path, model, optimizer)
    model.eval()

    images = []

    for i, angle in tqdm(enumerate(np.linspace(0.0, 360, 120, endpoint=False))):
        rgb_map, _ = render_image(model, poses, camera_info, args, i)
        image_values = rgb_map[..., :3]
        image_values = image_values.reshape((8,8,3))
        image_values = image_values.permute(2, 0, 1)

        image = np.array(torchvision.transforms.ToPILImage()(image_values.detach().cpu()))
        print("Image shape1: ", image.shape)
        # image = np.moveaxis(image, [-1], [0])
        # print("Image shape: ", image.shape)
        imageio.imwrite(f"output/image_{i}.png", image)

        images.append(image)


    imageio.mimwrite("gif.mp4", images, fps=30, quality=7, macro_block_size=None)


def main(args):
    # load data
    print("Loading data...")
    camera_info, images, poses = loadDataset(args.data_path, args.mode)
    # images_val, poses_val, camera_info_val = loadDataset(args.data_path, 'val')
    # images_test, poses_test, camera_info_test = loadDataset(args.data_path, 'test')

    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
        # validate(model, images_val, poses_val, camera_info_val, args)
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./data/lego",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=6,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--batchsize',default=1024*4,help="batchsize")
    parser.add_argument('--n_sample',default=8,help="number of sample per ray")
    parser.add_argument('--n_rays',default=64,help="number of rays")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./checkpoints/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")

    args = parser.parse_args()  # Parse the arguments
    
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    logger.info("#"*30)
    logger.info("####### PROGRAM START ########")
    logger.info("#"*30)

    main(args)
