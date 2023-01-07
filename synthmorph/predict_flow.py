import os
import numpy as np
import nibabel as nib
from voxelmorph.torch.networks import VxmDense
from nibabel.processing import conform
import torch

CASE = 'case_004'
# Load preprocessed data (scaled between 0 and 1 and with the moving data in the space of the fixed one)
fixed = nib.load(f"processed_data/copd_scans/scans/{CASE}_insp.nii.gz")

moving = nib.load(f"processed_data/copd_scans/scans/{CASE}_exp.nii.gz")
# print(moving.shape, moving.max(), moving.min())
# These data are in my local computer but any data could be used to perform the same analysis.
# It only needs to be scaled and set in a common space

# Load the PyTorch model and specify the device
device = 'cpu'
pt_model_inference = torch.load('/home/mira1/vlex_mira/maia-mira/models/weights/torch_shapes.pt')
pt_model_inference.bidir = True
pt_model_inference.eval()

# Prepare the data for inference
data_moving = np.expand_dims(moving.get_fdata().squeeze(), axis=(0, -1)).astype(np.float32)
data_fixed = np.expand_dims(fixed.get_fdata().squeeze(), axis=(0, -1)).astype(np.float32)

data_moving = data_moving/np.max(data_moving)
data_fixed  = data_fixed/np.max(data_fixed)

# Set up tensors and permute for inference
input_moving = torch.from_numpy(data_moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed = torch.from_numpy(data_fixed).to(device).float().permute(0, 4, 1, 2, 3)


with torch.no_grad():
    # Predict using PyTorch model
    moved, pos_flow, neg_flow= pt_model_inference(input_moving, input_fixed, registration=True)

moved_data = moved[0][0].detach().numpy()
moved_nifti = nib.Nifti1Image(moved_data, fixed.affine)
nib.save(moved_nifti, f"/home/mira1/vlex_mira/maia-mira/synthmorph/results/{CASE}_registered_exp2insp.nii.gz")
np.save(f"/home/mira1/vlex_mira/maia-mira/synthmorph/results/{CASE}_pos_flow.npy", pos_flow)
np.save(f"/home/mira1/vlex_mira/maia-mira/synthmorph/results/{CASE}_neg_flow.npy", neg_flow)