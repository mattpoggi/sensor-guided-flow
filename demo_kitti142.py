import sys

sys.path.append("core")

from PIL import Image
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import torch.nn as nn
import cv2

from core import datasets
from core.utils import frame_utils

from core.qraft import QRAFT
from core.utils.utils import InputPadder, forward_interpolate
from utils import *

import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer

lib = cdll.LoadLibrary("external/guided_flow/libguide.so")
build_guide = lib.build_guide

@torch.no_grad()
def validate_kitti(image1, image2, flow_gt, valid_gt, hints, valid_hints, name, model, iters=24, guided=False):
    """ Peform validation using the KITTI 142 split """
        
    image1 = image1[None].cuda()
    image2 = image2[None].cuda()

    padder = InputPadder(image1.shape, mode="kitti")
    image1, image2 = padder.pad(image1, image2)

    flow_pad = flow_gt.unsqueeze(0)
    valid_pad = valid_gt.unsqueeze(0).unsqueeze(0)
    flow_pad, valid_pad = padder.pad(flow_pad, valid_pad)

    hints_pad = hints.unsqueeze(0)
    valid_hints_pad = valid_hints.unsqueeze(0).unsqueeze(0)
    hints_pad, valid_hints_pad = padder.pad(hints_pad, valid_hints_pad)            

    hints = F.upsample(hints_pad, [hints_pad.size()[2]//4, hints_pad.size()[3]//4], mode='nearest') / 4
    valid_hints =  F.upsample(valid_hints_pad, [valid_hints_pad.size()[2]//4, valid_hints_pad.size()[3]//4], mode='nearest')
    hints = hints * valid_hints

    if guided:

        b, _, h, w = hints.shape
        k, c = 10., 1.
        guide_grid = np.zeros(b*w*h*w*h).astype(np.float32) # call here the build_guide function
        flow_x = hints[:,0,:,:].cpu().detach().numpy().reshape(-1)
        flow_y = hints[:,1,:,:].cpu().detach().numpy().reshape(-1)
        flow_valid = valid_hints.cpu().detach().numpy().reshape(-1)
        build_guide( c_void_p(flow_x.ctypes.data), c_void_p(flow_y.ctypes.data), c_void_p(flow_valid.ctypes.data), c_void_p(guide_grid.ctypes.data), c_int(b), c_int(h), c_int(w), c_float(k), c_float(c))
        guide_grid = torch.from_numpy(guide_grid.reshape(b, h, w, 1, h, w))
        guide_grid = guide_grid.cuda()

    else:
        guide_grid = None

    _, flow_predictions = model(image1, image2, iters=iters, test_mode=True, guide=guide_grid)
    flow = padder.unpad(flow_predictions[0]).cpu()
    del guide_grid

    epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
    mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
    fmag = torch.sum(flow ** 2, dim=0).sqrt()

    epe = epe.view(-1)
    mag = mag.view(-1)
    val = (valid_gt).view(-1) >= 0.5
    out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
    torch.cuda.empty_cache()

    return epe[val].cpu().numpy(), out[val].cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="select model: C, CT, CTS or CTK", default='CTK')
    parser.add_argument("--out_dir", type=str, help="path where to save outputs", default="results")
    parser.add_argument("--guided", action="store_true", help="enable guided OF")
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.cuda.synchronize()
    torch.no_grad()

    select_model = 'qraft'
    if args.guided:
        select_model = 'guided-qraft'
    checkpoint = 'weights/%s/%s.pth'%(select_model, args.model)
    model = torch.nn.DataParallel(QRAFT(args), device_ids=[0])
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cuda')))
    model.cuda()
    model.eval()

    if args.out_dir is not None and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with torch.no_grad():
        val_dataset = datasets.KITTI_142()
        pbar = tqdm.tqdm(total=len(val_dataset))
        epe_list = []
        fl_list = []
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt, hints, valid_hints, name = val_dataset[val_id]
            if os.path.exists("%s/%s_epe.npy"%(args.out_dir,name)) and os.path.exists("%s/%s_fl.npy"%(args.out_dir,name)):
                epe_list.append( np.load("%s/%s_epe.npy"%(args.out_dir,name)).mean() )
                fl_list.append( np.load("%s/%s_fl.npy"%(args.out_dir,name)) )
                pbar.update(1)
                continue
            epe, fl = validate_kitti(image1, image2, flow_gt, valid_gt, hints, valid_hints, name, model.module, root=args.val_root, output_path=args.out_dir, guided=args.guided, split142=("142" in args.dataset))
            np.save("%s/%s_epe.npy"%(args.out_dir,name), epe)
            np.save("%s/%s_fl.npy"%(args.out_dir,name), fl)
            
            epe_list.append(epe.mean())
            fl_list.append(fl)
            pbar.update(1)

    epe_list = np.array(epe_list)
    fl_list = np.concatenate(fl_list)
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(fl_list)        
    print("Validation KITTI: %.2f, %.2f" % (epe, f1))