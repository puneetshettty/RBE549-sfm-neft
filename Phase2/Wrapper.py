import glob
import json
import os
import math
import random

import imageio.v3 as iio
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from log_setup import logger
from NeRFModel import *

model = NeRFmodel()
positional_encoding = model.position_encoding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


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
    x = torch.linspace(0, H-1, H)
    y = torch.linspace(0, W-1, W)
    x, y = torch.meshgrid(x, y) 

    x = x.T
    y = y.T

    x = (x - H * 0.5) / focal
    y = (y - W * 0.5) / focal

    ray_direction = torch.stack([x, -y, -torch.ones_like(x)], -1)

    Rotation = pose[:3, :3]
    Translation = pose[:3, -1]

    directions = ray_direction[..., None, :]
    ray_directions = torch.sum(directions * Rotation, -1)
    ray_origins = torch.broadcast_to(Translation, ray_directions.shape)

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
    samples = torch.linspace(near, far, N_samples)
    rays = ray_origins[..., None, :] + t[..., None] * ray_directions[..., None, :]
    points = rays.reshape(-1, 3)
    points = positional_encoding(points, num_encoding_functions=4, include_input=True, log_sampling=True)


    return points, samples

import os  # Import the "os" module to use its functions

def generateBatch(raypoints, batch_size):
    """
    Input:
        raypoints: sampled points
        batch_size: number of rays per batch
    Outputs:
        batched points
    """
    inputs = []  # Define the "inputs" variable as an empty list
    for i in range(0, len(raypoints), batch_size):
        inputs.extend(raypoints[i: i + batch_size])  # Append the batched points to the "inputs" list

    return inputs
     

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
        radiance_field: rgb map from model
        ray_directions: direction of input rays
        depth_values: number of samples
    Outputs:
        rgb_map: rgb color of n depth values
        depth_map: depth map
        acc_map: accumulated map
    """


    sigma_a = F.relu(radiance_field[...,3])       #volume density
    # print("sigma", sigma_a.shape)
    rgb = torch.sigmoid(radiance_field[...,:3])    #color value at nth depth value
    # print("rgb", rgb.shape)
    one_e_10 = torch.tensor([1e10], dtype = ray_directions.dtype, device = ray_directions.device)
    # print("one_e_10", one_e_10.shape)
    dists = torch.cat((depth_values[...,1:] - depth_values[...,:-1], one_e_10.expand(depth_values[...,:1].shape)), dim = -1)
    # print("dists", dists.shape)
    alpha = 1. - torch.exp(-sigma_a * dists)       
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)     #transmittance
    rgb_map = (weights[..., None] * rgb).sum(dim = -2)          #resultant rgb color of n depth values
    depth_map = (weights * depth_values).sum(dim = -1)
    acc_map = weights.sum(-1)

    return rgb_map, depth_map, acc_map


    

def loss(groundtruth, prediction):
    loss = torch.mean((groundtruth - prediction)**2)
    return loss


def train(images, poses, camera_info, args):
    writer = SummaryWriter(args.logs_path)
    dataset = loadDataset(args.data_path, 'train')

    model = NeRFmodel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate)

    

def test(images, poses, camera_info, args):
    pass


def main(args):
    # load data
    print("Loading data...")
    images, poses, camera_info = loadDataset(args.data_path, args.mode)

    if args.mode == 'train':
        print("Start training")
        train(images, poses, camera_info, args)
        images_val, poses_val, camera_info_val = loadDataset(args.data_path, 'val')
    elif args.mode == 'test':
        print("Start testing")
        args.load_checkpoint = True
        test(images, poses, camera_info, args)

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./Phase2/Data/leg2o/",help="dataset path")
    parser.add_argument('--mode',default='train',help="train/test/val")
    parser.add_argument('--lrate',default=5e-4,help="training learning rate")
    parser.add_argument('--n_pos_freq',default=6,help="number of positional encoding frequencies for position")
    parser.add_argument('--n_dirc_freq',default=4,help="number of positional encoding frequencies for viewing direction")
    parser.add_argument('--n_rays_batch',default=32*32*4,help="number of rays per batch")
    parser.add_argument('--n_sample',default=64,help="number of sample per ray")
    parser.add_argument('--max_iters',default=10000,help="number of max iterations for training")
    parser.add_argument('--logs_path',default="./logs/",help="logs path")
    parser.add_argument('--checkpoint_path',default="./Phase2/example_checkpoint/",help="checkpoints path")
    parser.add_argument('--load_checkpoint',default=True,help="whether to load checkpoint or not")
    parser.add_argument('--save_ckpt_iter',default=1000,help="num of iteration to save checkpoint")
    parser.add_argument('--images_path', default="./image/",help="folder to store images")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    logger.info("#"*30)
    logger.info("####### PROGRAM START ########")
    logger.info("#"*30)

    main(args)