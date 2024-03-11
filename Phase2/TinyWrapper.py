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
from TinyNeRFModel import *
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
    randomize = True
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
                             encoding_function, get_minibatches_function, model):
  
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
    batches = get_minibatches_function(encoded_points, chunksize=1024)
    predictions = []
    for batch in batches:
        batch = batch.to(torch.float32)
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)
    print(torch.cuda.mem_get_info())

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

    frames = json_data['frames']
    images = [iio.imread(os.path.join(data_path, frame['file_path']+'.png')) for frame in frames]
    poses = np.array([np.array(frame['transform_matrix']) for frame in frames])

    height, width = images[0].shape[:2]
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
    
    camera_matrix = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]])

    return (width, height, camera_matrix, focal), images, poses

def load_latest_model(chk_path, model, optimizer):
    list_of_files = glob.glob(os.path.join(chk_path, '*'))
    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        checkpoint = torch.load(latest_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def validate(model, image, poses, camera_info, args, idx):
    rgb_map, ray_idx = render_image(model, poses, camera_info, args, idx)
    # Calculate the loss
    target_img = image[ray_idx[:,0], ray_idx[:,1], :]
    return loss_function(rgb_map[..., :3], target_img[..., :3])


def train(images, poses, camera_info, args):
    # Camera extrinsics (poses)
    tform_cam2world = poses
    tform_cam2world = torch.from_numpy(tform_cam2world).to(device)
    # Focal length (intrinsics)
    width, height, camera_matrix, focal_length = camera_info
    focal_length = torch.from_numpy(np.array(focal_length)).to(device)

    # Near and far clipping thresholds for depth values.
    near_thresh = 2.
    far_thresh = 6.
    print(images.__len__())

    # Hold one image out (for test).
    testimg, testpose = images[99], tform_cam2world[99]
    testimg = torch.from_numpy(testimg).to(device)
    print(np.shape(images))
    images = np.array(images)

    # Map images to device
    images = torch.from_numpy(images[:95, ..., :3]).to(device)
    print(width, height, testpose, focal_length )

    """
    Parameters for TinyNeRF training
    """

    # Number of functions used in the positional encoding (Be sure to update the 
    # model if this number changes).
    num_encoding_functions = 6
    # Specify encoding function.
    encode = lambda x: positional_encoding(x, num_encoding_functions=num_encoding_functions)
    # Number of depth samples along each ray.
    depth_samples_per_ray = 8

    # Chunksize (Note: this isn't batchsize in the conventional sense. This only
    # specifies the number of rays to be queried in one go. Backprop still happens
    # only after all rays from the current "bundle" are queried and rendered).
    chunksize = 8  # Use chunksize of about 4096 to fit in ~1.4 GB of GPU memory.

    # Optimizer parameters
    lr = 5e-3
    num_iters = 1000

    # Misc parameters
    display_every = 100  # Number of iters after which stats are displayed

    """
    Model
    """
    model = TinyNeRFmodel(num_encoding_functions=num_encoding_functions)
    model.to(device)

    """
    Optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    """
    Train-Eval-Repeat!
    """

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    for i in range(num_iters):

        # Randomly pick an image as the target.
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length,
                                                target_tform_cam2world, near_thresh,
                                                far_thresh, depth_samples_per_ray,
                                                encode, get_minibatches, model)

        # Compute mean-squared error between the predicted and target images. Backprop!
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Display images/plots/stats
        if i % display_every == 0:
            # Render the held-out view
            rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length,
                                                    testpose, near_thresh,
                                                    far_thresh, depth_samples_per_ray,
                                                    encode, get_minibatches, model)
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            print("Loss:", loss.item())
            psnr = -10. * torch.log10(loss)
            
            psnrs.append(psnr.item())
            iternums.append(i)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(rgb_predicted.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.show()

            print('Done!')

    
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
