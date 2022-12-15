"""Transforms keypoints using the negative flow field
and computes the TRE between the transformed keypoints and the fixed ones
before and after registration
."""

import os
import numpy as np
import nibabel as nib
from voxelmorph.torch.networks import VxmDense
from nibabel.processing import conform
import torch
import pandas as pd
from voxelmorph.tf.utils import point_spatial_transformer
import tensorflow as tf
from synthmorph.utils import compute_TRE

# Load preprocessed data (scaled between 0 and 1 and with the moving data in the space of the fixed one)
fixed = nib.load("processed_data/copd_scans/scans/case_001_insp.nii.gz")
fixed_a = fixed.get_fdata()
moving = nib.load("processed_data/copd_scans/scans/case_001_exp.nii.gz")
moving_a = moving.get_fdata()
kps = pd.read_csv("processed_data/copd_scans/keypoints/case_001.csv", header=None).values

    
moving_kps = kps[:, [0, 1, 2]]
fixed_kps = kps[:, [3, 4, 5]]

pos_flow = np.load("pos_flow.npy")
neg_flow = np.load("neg_flow.npy")

print(kps)

# # print(moving.shape, moving.max(), moving.min())
# # These data are in my local computer but any data could be used to perform the same analysis.
# # It only needs to be scaled and set in a common space

# Load the PyTorch model and specify the device
pt_model_inference = torch.load('pt_smshapes.pt')
pt_model_inference.eval()

# # Prepare the data for inference
# data_moving = np.expand_dims(moving.get_fdata().squeeze(), axis=(0, -1)).astype(np.float32)
# data_fixed = np.expand_dims(fixed.get_fdata().squeeze(), axis=(0, -1)).astype(np.float32)

# # Set up tensors and permute for inference
# input_moving = torch.from_numpy(data_moving).to(device).float().permute(0, 4, 1, 2, 3)
# input_fixed = torch.from_numpy(data_fixed).to(device).float().permute(0, 4, 1, 2, 3)

annotations = moving_kps[:, [0, 1, 2]]
annotations = annotations[np.newaxis, ...]

print(neg_flow.shape)
neg_flow_reshaped = np.moveaxis(neg_flow, [1], [4])
print(neg_flow_reshaped.shape)
# warp annotations
data = [tf.convert_to_tensor(f, dtype=tf.float32) for f in [annotations, neg_flow_reshaped]]
moving_transformed = point_spatial_transformer(data)[0, ...].numpy()

spc = [1.44, 1.44, 3.16] # USE ORIGINAL SPACING
print('Before: ', compute_TRE(moving_kps, fixed_kps, spc), )
print('After: ', compute_TRE(moving_transformed, fixed_kps, spc) )