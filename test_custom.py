import sys
import argparse
import os
import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

from model import fetch_model
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt, coords_grid, bilinear_sampler
from utils.frame_utils import writeFlow

from scipy.interpolate import griddata

from dataloader.flow.chairs import FlyingChairs
from dataloader.flow.sintel import MpiSintel
from dataloader.flow.kitti import KITTI
from dataloader.flow.spring import Spring
from dataloader.stereo.tartanair import TartanAir

from inference_tools import InferenceWrapper, AverageMeter

from xv_file import scan_files, get_file_name

def warp_with_flow(image, flow):
    N, _, H, W = image.shape
    coords2 = coords_grid(N, H, W, device=image.device).permute(0, 2, 3, 1)
    coords2 = coords2 + flow.permute(0, 2, 3, 1)
    warped_image = bilinear_sampler(image, coords2)
    return warped_image.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    cv2.imwrite(name, overlay)

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

@torch.no_grad()
def demo_data(name, args, model, image1, image2, flow_gt, outpath_flo, outpath_vis, outpath_heat, filename, valid=None, tiling=False):
    #path = f"demo/{name}/{args.name}/"
    #os.makedirs(path, exist_ok=True) #os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    #cv2.imwrite(f"{path}image1.jpg", cv2.cvtColor(image1[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    #cv2.imwrite(f"{path}image2.jpg", cv2.cvtColor(image2[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    #flow_gt_vis = flow_to_image(flow_gt[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    #cv2.imwrite(f"{path}gt.jpg", flow_gt_vis)
    output = model.calc_flow(image1, image2)
    i = len(output['flow']) - 1
    #for i in range():
    flow = output['flow'][i]
    flow_cpu = flow[0].permute(1, 2, 0).cpu().numpy()
    writeFlow(os.path.join(outpath_flo, f'{filename}.flo'), flow_cpu)   #(f"{path}flow_{i:06d}.flo", flow_cpu)
    flow_vis = flow_to_image(flow_cpu, convert_to_bgr=True)
    cv2.imwrite(os.path.join(outpath_vis, f'{filename}.jpg'), flow_vis)#f"{path}flow_{i}.jpg", flow_vis)
    #diff = flow_gt - flow
    #diff_vis = flow_to_image(diff[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    #cv2.imwrite(f"{path}error_{i}.jpg", diff_vis)
    if 'info' in output:
        info = output['info'][i]
        heatmap = get_heatmap(info, args)
        #f"{path}heatmap_{i}.jpg"
        vis_heatmap(os.path.join(outpath_heat, f'{filename}.jpg'), image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())
    if valid is None:
        N, _, H, W = flow.shape
        valid = torch.ones((N, H, W), device=flow.device)
    else:
        valid = valid.to(flow.device)
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    epe = (epe * valid).sum() / valid.sum()
    #print(f"EPE_step{i}: {epe.cpu().item()}")

@torch.no_grad()
def demo_chairs(model, args, device=torch.device('cuda')):
    dataset = FlyingChairs(split='training')
    image1, image2, flow_gt, _ = dataset[150]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('chairs', args, model, image1, image2, flow_gt)

def demo_sintel(model, args, device=torch.device('cuda')):
    dstype = 'final'
    dataset = MpiSintel(split='training', dstype=dstype)
    image1, image2, flow_gt, valid = dataset[400]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    valid = valid[None].to(device)
    demo_data('sintel', args, model, image1, image2, flow_gt, valid=valid, tiling=False)

def demo_kitti(model, args, device=torch.device('cuda')):
    dataset = KITTI(split='training')
    image1, image2, flow_gt, valid = dataset[100]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    valid = valid[None].to(device)
    demo_data('kitti', args, model, image1, image2, flow_gt, valid=valid, tiling=False)

@torch.no_grad()
def demo_spring(model, args, device=torch.device('cuda'), split='train'):
    dataset = Spring(split='val')
    idx = 175
    if split == 'train' or split == 'val':
        image1, image2, flow_gt, _ = dataset[idx]
    else:
        image1, image2,  _ = dataset[idx]
        h, w = image1.shape[1:]
        flow_gt = torch.zeros((2, h, w))

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('spring', args, model, image1, image2, flow_gt, tiling=False)

@torch.no_grad()
def demo_tartanair(model, args, device=torch.device('cuda')):
    dataset = TartanAir()
    image1, image2, flow_gt, valid = dataset[289992]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('tartanair', args, model, image1, image2, flow_gt, valid=valid)

@torch.no_grad()
def demo_custom(model, args, filename1, filename2, outpath_flo, outpath_vis, outpath_heat, device=torch.device('cuda')):
    image1 = cv2.imread(filename1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.imread(filename2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    #image1 = cv2.resize(image1, None, fx=0.5, fy=0.5)
    #image2 = cv2.resize(image2, None, fx=0.5, fy=0.5)

    image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
    H, W = image1.shape[1:]
    flow_gt = torch.zeros([2, H, W], device=device)
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)

    demo_data('custom_downsample', args, model, image1, image2, flow_gt, outpath_flo, outpath_vis, outpath_heat, get_file_name(filename1)[1])

def main():
    #& C:/Users/jdibe/miniconda3/envs/waft/python.exe d:/jcds/Documents/GitHub/WAFT/test_custom.py --cfg config/a1/tar-c-t-custom.json --ckpt checkpoints/tar-c-t.pth --dataset custom
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--ckpt', help='checkpoint path', required=True, type=str)
    parser.add_argument('--dataset', help='dataset to evaluate on', choices=['chairs', 'sintel', 'spring', 'tartanair', 'kitti', 'custom'], required=True, type=str)
    parser.add_argument('--scale', help='scale factor for input images', default=0.0, type=float)
    args = parse_args(parser)
    model = fetch_model(args)
    load_ckpt(model, args.ckpt)
    model = model.cuda()
    model.eval()
    wrapped_model = InferenceWrapper(model, scale=args.scale, train_size=args.image_size, pad_to_train_size=False, tiling=False)


    
    #files_l = sorted(scan_files('C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\X_etna_l'))
    #files_r = sorted(scan_files('C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\X_etna_r'))

    files_l = sorted(scan_files('C:\\Users\\jdibe\\OneDrive\\Desktop\\hl2\\img'))

    files_f1 = zip(files_l, files_l[1:])
    files_f2 = zip(files_l, files_l[2:])
    #files_d0 = zip(files_l, files_r)

    for filename1, filename2 in files_f1:
        print(f'{filename1} -> {filename2}')
        demo_custom(wrapped_model, args, filename1, filename2, 'C:\\Users\\jdibe\\OneDrive\\Desktop\\hl2\\flow', 'C:\\Users\\jdibe\\OneDrive\\Desktop\\hl2\\vis_flow', 'C:\\Users\\jdibe\\OneDrive\\Desktop\\hl2\\heat_flow')
        #break

    for filename1, filename2 in files_f2:
        print(f'{filename1} -> {filename2}')
        demo_custom(wrapped_model, args, filename1, filename2, 'C:\\Users\\jdibe\\OneDrive\\Desktop\\hl2\\flow2', 'C:\\Users\\jdibe\\OneDrive\\Desktop\\hl2\\vis_flow2', 'C:\\Users\\jdibe\\OneDrive\\Desktop\\hl2\\heat_flow2')
        #break

    #for filename1, filename2 in files_d0:
    #    print(f'{filename1} -> {filename2}')
    #    demo_custom(wrapped_model, args, filename1, filename2, 'C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\disp', 'C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\vis_disp', 'C:\\Users\\jdibe\\OneDrive\\Desktop\\X_etna\\heat_disp')
    #    #break




    
    
    
    #if args.dataset == 'chairs':
    #    demo_chairs(wrapped_model, args)
    #elif args.dataset == 'sintel':
    #    demo_sintel(wrapped_model, args)
    #elif args.dataset == 'spring':
    #    demo_spring(wrapped_model, args)
    #elif args.dataset == 'tartanair':
    #    demo_tartanair(wrapped_model, args)
    #elif args.dataset == 'kitti':
    #    demo_kitti(wrapped_model, args)

if __name__ == '__main__':
    main()