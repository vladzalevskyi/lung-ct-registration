
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from numba import njit

def resample2target(moving, target):
    """Resamples moving image to target image"""

    return sitk.Resample(moving, target.GetSize(),
                                    sitk.Transform(), 
                                    sitk.sitkLinear,
                                    target.GetOrigin(),
                                    target.GetSpacing(),
                                    target.GetDirection(),
                                    0,
                                    target.GetPixelID())


def print_img_info(selected_image, title='Train image:'):
    print(title)
    print('origin: ' + str(selected_image.GetOrigin()))
    print('size: ' + str(selected_image.GetSize()))
    print('spacing: ' + str(selected_image.GetSpacing()))
    print('direction: ' + str(selected_image.GetDirection()))
    print('pixel type: ' + str(selected_image.GetPixelIDTypeAsString()))
    print('number of pixel components: ' + str(selected_image.GetNumberOfComponentsPerPixel()))


# a simple function to plot an image
def plot1(fixed, title='', slice=128, figsize=(12, 12)):
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.imshow(sitk.GetArrayFromImage(fixed)[slice, :, :], cmap='gray', origin='lower')
    axs.set_title(title, fontdict={'size':26})
    axs.axis('off')
    plt.tight_layout()
    plt.show()
    
# a simple function to plot 3 images at once
def plot3(fixed, moving, transformed, labels=['Fixed', 'Moving', 'Moving Transformed'], slice=128):
    fig, axs = plt.subplots(1, 3, figsize=(24, 12))
    axs[0].imshow(sitk.GetArrayFromImage(fixed)[slice, :, :], cmap='gray', origin='lower')
    axs[0].set_title(labels[0], fontdict={'size':26})
    axs[0].axis('off')
    axs[1].imshow(sitk.GetArrayFromImage(moving)[slice, :, :], cmap='gray', origin='lower')
    axs[1].axis('off')
    axs[1].set_title(labels[1], fontdict={'size':26})
    axs[2].imshow(sitk.GetArrayFromImage(transformed)[slice, :, :], cmap='gray', origin='lower')
    axs[2].axis('off')
    axs[2].set_title(labels[2], fontdict={'size':26})
    plt.tight_layout()
    plt.show()
    
def padd_images_to_max(img1, img2):
    """Padds images so that they have the same size.
    
    Zero paddig is used on the right and bottom side of the image
    so that image with smallest dimensions is padded to the size of the largest one."""
    padding_filter = sitk.ConstantPadImageFilter()

    for dim in range(len(img1.GetSize())):
        padding_array = [0, 0, 0]
        padding_size = img1.GetSize()[dim] - img2.GetSize()[dim]
        padding_array[dim] = abs(padding_size)
        
        if img1.GetSize()[dim] > img2.GetSize()[dim]:
            img2 = padding_filter.Execute(img2, (0, 0, 0), padding_array, 0)
        else:
            img1 = padding_filter.Execute(img1, (0, 0, 0), padding_array, 0)
    return img1, img2

def register_image(fixed_image, moving_image, transformParameterMap=None, interpolator='nn'):
    """Perform registration of the moving image to fixed image. 
    
    Either learn the registration and return the result and transformParameterMap
    or use the one passed as an argument

    Args:
        fixed_image (sitk.Image): Fixed image
        moving_image (sitk.Image): Moving image
        transformParameterMap (SimpleITK.SimpleITK.ParameterMap, optional): Parameter map used for registration.Defaults to None.
            If None, the registration is learned and the learned parameter map is returned.
            Otherwise, uses the map passed to perform the transformation.
        interpolator (str, optional): If 'nn' changes the interpolator in the transformParameterMap
            to FinalNearestNeighborInterpolator. Otherwise, uses the one from the map. 
            Defaults to 'nn'.

    Returns:
        (transformed image (sitk.Image), transformParameterMap)
    """
    fixed_image, moving_image = padd_images_to_max(fixed_image, moving_image)
    if transformParameterMap is None:
        elastixImageFilter_non_rigid = sitk.ElastixImageFilter()
        elastixImageFilter_non_rigid.SetFixedImage(fixed_image)
        elastixImageFilter_non_rigid.SetMovingImage(moving_image)

        elastixImageFilter_non_rigid.Execute()
        transformParameterMap = elastixImageFilter_non_rigid.GetTransformParameterMap()
        return elastixImageFilter_non_rigid.GetResultImage(), transformParameterMap
        
    else:
        transformixImageFilter = sitk.TransformixImageFilter()
        # change interpolator to NN for label
        if interpolator == 'nn':
            for transfrom in transformParameterMap:
                transfrom['ResampleInterpolator'] = ('FinalNearestNeighborInterpolator',)

        transformixImageFilter.SetTransformParameterMap(transformParameterMap)
        transformixImageFilter.SetMovingImage(moving_image)
        transformixImageFilter.Execute()
        return transformixImageFilter.GetResultImage(), transformParameterMap
        
def register_image_w_mask(fixed_image, fixed_mask, moving_image, moving_mask, transformParameterMap=None, interpolator='nn'):
    """Perform registration of the moving image to fixed image. 
    
    Either learn the registration and return the result and transformParameterMap
    or use the one passed as an argument

    Args:
        fixed_image (sitk.Image): Fixed image
        moving_image (sitk.Image): Moving image
        transformParameterMap (SimpleITK.SimpleITK.ParameterMap, optional): Parameter map used for registration.Defaults to None.
            If None, the registration is learned and the learned parameter map is returned.
            Otherwise, uses the map passed to perform the transformation.
        interpolator (str, optional): If 'nn' changes the interpolator in the transformParameterMap
            to FinalNearestNeighborInterpolator. Otherwise, uses the one from the map. 
            Defaults to 'nn'.

    Returns:
        (transformed image (sitk.Image), transformParameterMap)
    """
    # fixed_image, moving_image = padd_images_to_max(fixed_image, moving_image)

    if transformParameterMap is None:
        elastixImageFilter_non_rigid = sitk.ElastixImageFilter()
        elastixImageFilter_non_rigid.SetFixedImage(fixed_image)
        elastixImageFilter_non_rigid.SetMovingImage(moving_image)
        elastixImageFilter_non_rigid.SetFixedMask(fixed_mask)
        elastixImageFilter_non_rigid.SetMovingMask(moving_mask)

        elastixImageFilter_non_rigid.Execute()
        transformParameterMap = elastixImageFilter_non_rigid.GetTransformParameterMap()
        return elastixImageFilter_non_rigid.GetResultImage(), transformParameterMap
        
    else:
        transformixImageFilter = sitk.TransformixImageFilter()
        # change interpolator to NN for label
        if interpolator == 'nn':
            for transfrom in transformParameterMap:
                transfrom['ResampleInterpolator'] = ('FinalNearestNeighborInterpolator',)

        transformixImageFilter.SetTransformParameterMap(transformParameterMap)
        transformixImageFilter.SetMovingImage(moving_image)
        transformixImageFilter.Execute()
        return transformixImageFilter.GetResultImage(), transformParameterMap
        


def tissue_model_segmentation(image, brain_mask, tissue_model):
    """
    Compute segmentation from a brain volume by mapping each gray level
    to a LUT of tissue models.

    Args:
        image (np.ndarray): 3D volume of brain
        brain_mask (np.ndarray): 3D volume (same size as image) with brain tissues GT != 0
        tissue_model (np.ndarray): LUT of tissue models, columns = [0 - CSF, 1 - WM, 2 - GM]

    Returns:
        result_discr (np.ndarray): Discrete segemented brain with labels 1 - CSF, 2 - WM, 3 - GM
        result_prb (np.ndarray): Tissue probabilities volumes. Last axis of array corresponds to
                                probability volumes: 0 - CSF, 1 - WM, 2 - GM
    """
    image_shape = image.shape
    result_discr = np.zeros((image_shape))
    result_prb = np.zeros((image_shape[0], image_shape[1], image_shape[2], 3))

    image_flat = image[brain_mask!= 0]

    # probs
    probs = np.apply_along_axis(lambda x: tissue_model[x,:], 0, np.uint16(image_flat))

    # discretized
    seg = np.argmax(probs, axis=1) + 1

    result_discr[brain_mask!=0] = seg

    for i in range(3):
        result_prb[brain_mask!=0,i] = probs[:,i]

    return result_discr, result_prb

def label_propagation(brain_mask, atlas_list):
    """
    Generate final predicted labels volume from a list of registered 
    probabilistic atlas. The atlasses must be previously registered to a target image.
    Also returns the list of atlasses as a np.ndarray for further usage.

    Args:
        brain_mask (np.ndarray): 3D volume (same size as target image) with brain tissues GT != 0
        atlas_list (List): List np.ndarray volumes, same size as target image

    Returns:
        result_discr (np.ndarray): Discrete segemented brain with labels 1 - CSF, 2 - WM, 3 - GM
        result_prb (np.ndarray): Tissue probabilities volumes. Last axis indexes of array corresponds to
                                probability volumes: 0 - CSF, 1 - WM, 2 - GM
    """
    
    image_shape = atlas_list[0].shape
    result_discr = np.zeros((image_shape))
    result_prb = np.zeros((image_shape[0], image_shape[1], image_shape[2], 3))
    probs = np.zeros(((brain_mask > 0).sum(),3))

    result_prb[:,:,:,0] = atlas_list[0]
    result_prb[:,:,:,1] = atlas_list[1]
    result_prb[:,:,:,2] = atlas_list[2]

    for i in range(3):
        probs[:,i] = result_prb[brain_mask != 0, i]
        result_prb[brain_mask == 0, i] = 0

    # discretized
    seg = np.argmax(probs, axis=1) + 1
    result_discr[brain_mask!=0] = seg

    return result_discr, result_prb

def get_landmarks_image(landmarks: np.ndarray, image_shape: tuple):
    """From a np.ndarray of points, create a labels image. Each point
    will have it's corresponding index in the list as label

    Args:
        landmarks (np.ndarray): list of points
        image_shape (tuple): shape of target image

    Returns:
        lm_img (np.ndarray): labels image
    """
    lm_img = np.zeros(image_shape)
    for i, (x, y, z) in enumerate(landmarks):
        lm_img[y, x, z] = i + 1
    return lm_img

def parse_points_reg(file_name):
    """Parse points file from registration

    Args:
        file_name (str): Path to the points file
        """
    with open(file_name, 'r') as f:
        lines = f.readlines()
        points = []
        for line in lines:
            points.append(list(line.split('[')[-1][:-2].strip().split(' ')))

    return np.asarray(points).astype(np.int16)

@njit
def get_label_coordinates(image):

    coord_list = []
    shape = image.shape
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if image[x,y,z] != 0:
                    coord_list.append([image[x, y, z], x, y, z])
    return coord_list

def compute_TRE(points_fixed, points_moved, voxel_spacing):
    """Compute Target Registration Error (TRE) between two sets of points.

    Args:
        points_fixed (np.ndarray): points in fixed space
        points_moved (np.ndarray): points in moved space

    Returns:
        mean TRE: average TRE over all points
        std TRE: standard deviation of TRE over all points
    """
    differences = (points_moved - points_fixed) * voxel_spacing
    distances = np.linalg.norm(differences, axis=1)
    return np.mean(distances), np.std(distances) 


def convert_nda_to_itk(nda: np.ndarray, itk_image: sitk.Image):
    """From a numpy array, get an itk image object, copying information
    from an existing one. It switches the z-axis from last to first position.

    Args:
        nda (np.ndarray): 3D image array
        itk_image (sitk.Image): Image object to copy info from

    Returns:
        new_itk_image (sitk.Image): New Image object
    """
    new_itk_image = sitk.GetImageFromArray(np.moveaxis(nda, -1, 0))
    new_itk_image.SetOrigin(itk_image.GetOrigin())
    new_itk_image.SetSpacing(itk_image.GetSpacing())
    new_itk_image.CopyInformation(itk_image)
    return new_itk_image

def convert_itk_to_nda(itk_image: sitk.Image):
    """From an itk Image object, get a numpy array. It moves the first z-axis
    to the last position (np.ndarray convention).

    Args:
        itk_image (sitk.Image): Image object to convert

    Returns:
        result (np.ndarray): Converted nda image
    """
    return np.moveaxis(sitk.GetArrayFromImage(itk_image), 0, -1)

def compute_image_TRE(lm_img: np.ndarray, lm_img_transformed: np.ndarray, voxel_spacing=(1, 1, 1)):
    """From two images of landmarks labels (each containing keypoints with consecutive labels),
     obtain the TRE considering voxel spacing.

    Args:
        lm_img (np.ndarray): Landmark image (fixed)
        lm_img_transformed (np.ndarray): Landmark image (moving transformed)
        voxel_spacing (tuple, optional): Voxel spacing. Defaults to (1, 1, 1).

    Returns:
        mean, std: Mean and standard deviation of the each landmark correspondance euclidean distance.
    """
    # get coordinates of landmarks
    coords_img = np.asarray(get_label_coordinates(lm_img))
    coords_img_transformed = np.asarray(get_label_coordinates(lm_img_transformed))

    # sort coordinates by label
    coords_img = coords_img[coords_img[:, 0].argsort()]
    coords_img_transformed = coords_img_transformed[coords_img_transformed[:, 0].argsort()]
    
    # compute difference
    difference = (coords_img[:, 1:] - coords_img_transformed[:, 1:]) * voxel_spacing

    # compute 3D euclidean distance
    distances = np.linalg.norm(difference, axis=1)

    return np.mean(distances), np.std(distances)