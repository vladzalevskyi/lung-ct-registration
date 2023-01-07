import os
import numpy as np
import nibabel as nib
from voxelmorph.torch.networks import VxmDense
from nibabel.processing import conform
import torch
# Load preprocessed data (scaled between 0 and 1 and with the moving data in the space of the fixed one)
fixed = nib.load("data/learn2reg/scans/case_001_insp.nii.gz")
fixed_scaled = conform(fixed,
                                      out_shape=(160, 160, 192),
                                      voxel_size=(1.25, 1.25, 1.25)).get_fdata()
fixed_scaled = (fixed_scaled - fixed_scaled.min())/(fixed_scaled.max() - fixed_scaled.min())


moving = nib.load("data/learn2reg/scans/case_001_exp.nii.gz")
moving_scaled = conform(moving,
                                      out_shape=(160, 160, 192),
                                      voxel_size=(1.25, 1.25, 1.25)).get_fdata()
moving_scaled = (moving_scaled - moving_scaled.min())/(moving_scaled.max() - moving_scaled.min())

# N.B.
# These data are in my local computer but any data could be used to perform the same analysis.
# It only needs to be scaled and set in a common space

# Load the PyTorch model and specify the device
device = 'cpu'
pt_model_inference = torch.load('pt_smshapes.pt')
pt_model_inference.eval()

# Prepare the data for inference
data_moving = np.expand_dims(moving_scaled[:160, :160, :192].squeeze(), axis=(0, -1)).astype(np.float32)
data_fixed = np.expand_dims(fixed_scaled[:160, :160, :192].squeeze(), axis=(0, -1)).astype(np.float32)

# Set up tensors and permute for inference
input_moving = torch.from_numpy(data_moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed = torch.from_numpy(data_fixed).to(device).float().permute(0, 4, 1, 2, 3)



# Predict using PyTorch model
moved, pos_flow = pt_model_inference(input_moving, input_fixed, registration=True)
moved_data = moved[0][0].detach().numpy()
moved_nifti = nib.Nifti1Image(moved_data, fixed.affine)
nib.save(moved_nifti, 'registered_data_pytorch.nii.gz')
nib.save(pos_flow, 'pos_flow.nii.gz')
