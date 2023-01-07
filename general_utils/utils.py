
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from numba import njit
from typing import List
from pathlib import Path
from skimage.exposure import equalize_adapthist
from lungmask import mask

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

def register_image_external(fixed_image: sitk.Image, fixed_mask: sitk.Image, moving_image: sitk.Image, 
                            moving_mask: sitk.Image, paramMaps: List, print_console = False):
    
    """Using simple elastix, register two images with masks (to focus only on that region)
    using external parameter Maps

    Args:
        fixed_image (sitk.Image): Fixed image
        fixed_mask (sitk.Image): Fixed image mask, binary
        moving_image (sitk.Image): Moving image
        moving_mask (sitk.Image): Moving image mask, binary
        paramMaps (List): List of parameter Maps
        print_console (bool, optional): Whether to print console Log. Defaults to False.

    Returns:
        result_image, final_param_map (tuple): Result image of the registration (transformed moving)
        and the transformation parameter map (including the deformation field, in case of non-rigid)
    """
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetNumberOfThreads(8)
    if not print_console:
        elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)

    if fixed_mask is not None:
        elastixImageFilter.SetFixedMask(fixed_mask)
    if moving_mask is not None:
        elastixImageFilter.SetMovingMask(moving_mask)

    if len(paramMaps) == 1:
        elastixImageFilter.SetParameterMap(paramMaps[0])
    else:
        for i, paramMap in enumerate(paramMaps):
            if i == 0:
                elastixImageFilter.SetParameterMap(paramMap)
            elastixImageFilter.AddParameterMap(paramMap)

    elastixImageFilter.Execute()
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    return elastixImageFilter.GetResultImage(), transformParameterMap

    
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
        
def register_image_w_mask(fixed_image, fixed_mask, moving_image, moving_mask, transformParameterMap=None, 
                            interpolator='nn', default_type='affine', print_console = False):
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
        default_type (str). ParameterMap default type. Can be either 'translation', 'affine', or 'bspline'.

    Returns:
        (transformed image (sitk.Image), transformParameterMap)
    """
    # fixed_image, moving_image = padd_images_to_max(fixed_image, moving_image)

    if transformParameterMap is None:
        
        elastixImageFilter = sitk.ElastixImageFilter()
        if not print_console:
            elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.SetFixedImage(fixed_image)
        elastixImageFilter.SetMovingImage(moving_image)
        elastixImageFilter.SetFixedMask(fixed_mask)
        elastixImageFilter.SetMovingMask(moving_mask)

        # if default_type == 'affine':
        #     elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
        #     print('Set default affine registration')
        # elif default_type == 'bspline':
        #     elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("bspline"))
        #     print('Set default b-spline registration')

        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap(default_type))
        print(f'Set default {default_type} registration')    

        elastixImageFilter.Execute()
        transformParameterMap = elastixImageFilter.GetTransformParameterMap()
        return elastixImageFilter.GetResultImage(), transformParameterMap
        
    else:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.LogToConsoleOff()
        # change interpolator to NN for label
        if interpolator == 'nn':
            for transfrom in transformParameterMap:
                transfrom['ResampleInterpolator'] = ('FinalNearestNeighborInterpolator',)

        transformixImageFilter.SetTransformParameterMap(transformParameterMap)
        transformixImageFilter.SetMovingImage(moving_image)
        transformixImageFilter.Execute()
        result_image = transformixImageFilter.GetResultImage()
        result_image.SetSpacing(moving_image.GetSpacing())
        return result_image, transformParameterMap

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

def normalize_copd_to_HU(copd_image: np.ndarray, itk_image: sitk.Image, slope = 1, intercept = -1024):
    
    # set fov to 0
    copd_image[copd_image == -2000] = 0

    # to HU units
    image_hu_nda = (copd_image * slope) + intercept

    return convert_nda_to_itk(image_hu_nda, itk_image)

def preprocess_image(image: sitk.Image):
    """Apply preprocessing: min-max normalization and clahe

    Args:
        image (sitk.Image): Image to preprocess

    Returns:
        sitk.Image: Processed image
    """
    image_nda = convert_itk_to_nda(image)
    image_nda[image_nda<0] = 0
    image_nda = image_nda/np.max(image_nda)
    image_nda = equalize_adapthist(image_nda)

    return convert_nda_to_itk(image_nda, image)

def segment_lungs(image: sitk.Image, output_path = None):
    """Segment lungs from a lung CT scan.

    Args:
        image (sitk.Image): Image to segment
        write_image (bool, optional): Whether to write the image in Disk. 
        Defaults to True.
        output_path (_type_, optional): Path with filename. Defaults to None.

    Returns:
        _type_: _description_
    """
    # normalize to HU
    image_nda = convert_itk_to_nda(image)
    image_HU = normalize_copd_to_HU(image_nda, image)

    # segment using U-Net
    segmentation = mask.apply(image_HU, force_cpu=True)
    segmentation = np.moveaxis((segmentation >= 1).astype(np.uint8), 0, -1)
    lung_mask_fixed = convert_nda_to_itk(segmentation, image)

    if output_path:
        sitk.WriteImage(lung_mask_fixed, str(output_path))

    return lung_mask_fixed

def save_pts_itk(points: np.ndarray, path: Path):
    """Generate the .pts file for transformation of points using
    simple itk's transformix object.

    Args:
        points (np.ndarray): array with points with shape=(300, 3)
        path (Path): path to the file, with .pts extension
    """
    with open(path, 'w') as f:
        f.write('index\n')
        f.write('300\n')
        for row in points:
            f.write(f' {row[0]}\t {row[1]}\t {row[2]}\t\n')

def save_pts(points: np.ndarray, path: Path):
    """Generate final txt file for TRE computation
    
    Args:
        points (np.ndarray): array with points with shape=(300, 3)
        path (Path): path to the file, with .txt extension
    """
    with open(path, 'w') as f:
        for row in points:
            f.write(f' {row[0]}\t {row[1]}\t {row[2]}\t\n')

def transform_points(moving_itk: sitk.Image, mov_param: sitk.ParameterMap, 
                    fixed_points_path: Path, output_dir: Path, train_case: str, save_points = True):
    """Process points to transform them using a transformation parameter object (itk). Returns 
    the np.ndarray with the points for further TRE computation. It also outputs the points as a .pts
    file for consecute registration transformations with elastix.

    Args:
        moving_itk (sitk.Image): Itk image to copy metadata
        mov_param (_type_): transformation parameters
        fixed_points_path (Path): Path to read the points that will be transformed.
        Usually fixed image keypoints
        output_dir (Path): Output directory where the "outputpoints.txt" will be written
    """
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()

    # set moving image to initialize object correctly
    transformixImageFilter.SetMovingImage(moving_itk)

    # set previously obtained transformation parameters
    transformixImageFilter.SetTransformParameterMap(mov_param)

    # set points to transform (inhale)
    transformixImageFilter.SetFixedPointSetFileName(str(fixed_points_path))
    transformixImageFilter.SetOutputDirectory(str(output_dir))
    transformixImageFilter.Execute()

    # from output file, parse the output points and compute TRE
    out_points_path = output_dir/'outputpoints.txt'
    pts_transformed = parse_points_reg(out_points_path)

    # save points in required format by itk for further steps
    if save_points:
        save_pts_itk(pts_transformed, output_dir/f'case_{train_case}_outputpoints.pts')
    
    return pts_transformed

def transform_points_test(moving_itk: sitk.Image, mov_param: sitk.ParameterMap, 
                    fixed_points_path: Path, output_dir: Path):
    """Process points to transform them using a transformation parameter object (itk). Returns 
    the np.ndarray with the points for further TRE computation. It also outputs the points as a .pts
    file for consecute registration transformations with elastix.

    Args:
        moving_itk (sitk.Image): Itk image to copy metadata
        mov_param (_type_): transformation parameters
        fixed_points_path (Path): Path to read the points that will be transformed.
        Usually fixed image keypoints
        output_dir (Path): Output directory where the "outputpoints.txt" will be written
    """
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()

    # set moving image to initialize object correctly
    transformixImageFilter.SetMovingImage(moving_itk)

    # set previously obtained transformation parameters
    transformixImageFilter.SetTransformParameterMap(mov_param)

    # set points to transform (inhale)
    transformixImageFilter.SetFixedPointSetFileName(str(fixed_points_path))
    transformixImageFilter.SetOutputDirectory(str(output_dir))
    transformixImageFilter.Execute()

    # from output file, parse the output points and compute TRE
    out_points_path = output_dir/'outputpoints.txt'
    pts_transformed = parse_points_reg(out_points_path)

    # save points in required format for TRE computation
    save_pts(pts_transformed, output_dir/f'transformedpoints.txt')
    
    return pts_transformed