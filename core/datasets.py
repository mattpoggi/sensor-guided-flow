# Data loading based on https://github.com/princeton-vl/RAFT

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from core.utils import frame_utils
from typing import List


class FlowDataset(data.Dataset):
    def __init__(self):

        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.flow_reading_func = None

        self.hints_list = []

    def __getitem__(self, index):

        flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        hints, valid_hints = frame_utils.readFlowKITTI(self.hints_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        hints = np.array(hints).astype(np.float32)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid).float()
        hints = torch.from_numpy(hints).permute(2, 0, 1).float()
        valid_hints = torch.from_numpy(valid_hints).float()

        return (
                img1,
                img2,
                flow,
                valid,
                hints,
                valid_hints,
                self.extra_info[index][0],
        )

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)

class KITTI_142(FlowDataset):
    def __init__(
        self, aug_params=None, root="data/", hints_folder='hints'):
        super(KITTI_142, self).__init__()

        root = osp.join(root, "training")
        self.hints_list = sorted(glob(osp.join(root, "%s/*_10.png"%hints_folder)))
        images1 = [f.replace("%s"%hints_folder, "image_2") for f in self.hints_list]
        images2 = [f.replace("_10", "_11") for f in images1]
        self.flow_list = [f.replace("%s"%hints_folder, "flow_occ") for f in self.hints_list]

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1].split(".")[0]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]