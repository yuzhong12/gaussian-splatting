#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
import os  # [新增] 用于文件路径判断
from PIL import Image  # [新增] 用于读取 Mask 图片

class Camera(nn.Module):
    # [新增] 在参数列表中增加 mask_path=None
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 mask_path=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        
        # 原始的 Alpha Channel 处理逻辑 (保留作为备用)
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))
            
        # [新增] 自定义 Mask 读取与处理逻辑
        self.gt_alpha_mask = None
        if mask_path is not None and os.path.exists(mask_path):
            # 1. 读取 Mask 图片
            mask_pil = Image.open(mask_path)
            # 2. 调整大小以匹配当前分辨率 (resolution)
            mask_pil = mask_pil.resize(image.size, Image.Resampling.BILINEAR)
            # 3. 转为 Tensor
            # 注意：PILtoTorch 会自动除以 255。如果你的 mask 像素值真的是 0 和 1 (几乎全黑)，
            # 我们需要特殊处理。如果是 0 和 255 (黑白)，PILtoTorch 后就是 0.0 和 1.0。
            temp_mask = PILtoTorch(mask_pil, resolution)[:1, ...] # 取单通道
            
            # 4. 值域校正：如果最大值 <= 1.1，说明原图可能就是 0/1 存储的，
            # PILtoTorch 除以 255 后会变得非常小，所以我们要乘回去，或者直接二值化。
            # 这里采用稳健的二值化方法：大于 0.5 (对应像素值128) 算 1，否则算 0。
            # 如果你的原图是严格的 0 和 1 像素值，建议直接 mask_pil 转 numpy 处理。
            # 鉴于你说只有 0 和 1，这里做一个兼容性处理：
            if torch.max(temp_mask) < 0.01: # 即使是 1/255 也是 0.0039，小于 0.01
                # 说明原图大概率是 0/1 像素值
                 self.gt_alpha_mask = (temp_mask > 0).float().to(self.data_device)
            else:
                # 说明原图是 0/255 (标准黑白图)
                self.gt_alpha_mask = (temp_mask > 0.5).float().to(self.data_device)
            
            # 用读取到的 mask 覆盖默认的 alpha_mask
            self.alpha_mask = self.gt_alpha_mask
        else:
            # 如果没有外部 mask 路径，检查原图是否自带 Alpha 通道 (4通道)
            if resized_image_rgb.shape[0] == 4:
                # 如果有，就用自带的 Alpha
                self.gt_alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
                # 同样做一个二值化处理，确保只有 0 和 1
                self.gt_alpha_mask[self.gt_alpha_mask >= 0.5] = 1.0
                self.gt_alpha_mask[self.gt_alpha_mask < 0.5] = 0.0
            else:
                # 如果既没外部 mask 也没自带 alpha，才全设为 1
                self.gt_alpha_mask = torch.ones_like(gt_image[0:1, ...]).to(self.data_device)       

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0
                

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        
        # [新增] 应用 Mask 到原图 (把背景变黑)，冗余
        self.original_image *= self.gt_alpha_mask
        
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

