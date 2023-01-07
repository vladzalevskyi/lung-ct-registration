import os

# Import the Keras model and copy the weights
from voxelmorph.tf.networks import VxmDense
tf_model = VxmDense.load('models/weights/shapes-dice-vel-3-res-8-16-32-256f.h5', input_model=None)
weights = tf_model.get_weights()

# import VoxelMorph with pytorch backend
import torch
from voxelmorph.torch.networks import VxmDense


# ---- CONVERT THE KERAS/TENSORFLOW H5 MODEL TO PYTORCH ---- #

# IMG_DIM = (512, 512, 96) # INPUT IMAGE DIMENSIONS (should be divisible by 32 because of 5 level unet)
IMG_DIM = (224, 224, 96)

# Build a Torch model and set the weights from the Keras model
reg_args = dict(
    inshape=IMG_DIM,
    int_steps=5,
    int_downsize=2,   # same as int_resolution=2, in keras model
    unet_half_res=True,  # same as svf_resolution=2, in keras model
    nb_unet_features=([256, 256, 256, 256], [256, 256, 256, 256, 256, 256])
)
# Create the PyTorch model
pt_model = VxmDense(**reg_args)

# Load the weights onto the PyTorch model
i = 0
i_max = len(list(pt_model.named_parameters()))
torchparam = pt_model.state_dict()

for k, v in torchparam.items():
    if i < i_max:
        
        print("{:20s} {}".format(k, v.shape))
        
        if k.split('.')[-1] == 'weight':
            # torchparam[k] = torch.tensor(weights[i].T)
            # only swap the last two dimensions fix from
            # https://github.com/voxelmorph/voxelmorph/issues/425
            torchparam[k]  = torch.movedim(torch.tensor(weights[i]), (-1, -2), (0, 1))
        else:
            torchparam[k] = torch.tensor(weights[i])
        
        i += 1

pt_model.load_state_dict(torchparam)

# Save the PyTorch model
torch.save(pt_model, 'models/weights/torch_shapes.pt')