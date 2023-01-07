""""
Preprocesses data for synthmorph registration



Considerations

Accurate registration requires the input images
to be min-max normalized, such that voxel intensities range
from 0 to 1, and to be resampled in the affine space of a reference image.

While you should not have to retrain this model to use
a different image shape, it is hard to predict what went
wrong without seeing your code. If the image shape is (2048, 2048),
for example, the shape of both model inputs should be (1, 2048, 2048, 1).
That is, the model expects a leading batch and a trailing feature dimension.
In contrast, we only specify the spatial dimensions in_shape = (2048, 2048) when
setting up the model in the demo. That being said, you will get best results when 
the test distribution of image characteristics is included in the training 
distribution (spatial scales, shapes, etc.).

The model weights are independent of the input image size.
However, you need to specify the image shape you want when
constructing the model, by passing inshape=(2048, 2048) to vxm.networks.VxmDense

"""

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

from utils import compute_TRE, parse_points_reg

processed_dir = Path('processed_data').mkdir(exist_ok=True)
data_dir = Path('data/')
min_max_filter = sitk.MinimumMaximumImageFilter()

def crop_sitkimage(image, lung_mask, dim=(192, 192, 192)):
    """Crops image to around cenre of mass of the mask
    
    Returns coordinates of the patch, cropping the image at center location
    with a given patch size. If the center is in the left or upper border shift
    the center and crop fixed size patch.
    Args:
        center (tuple): (x coordinate, y coordinate)
        image_shape (tuple): shape of image to crop patches from
        dim (int, optional): patch size
    Returns:
        cropped sitk image
    """
    center = ndimage.center_of_mass(sitk.GetArrayFromImage(lung_mask).T)
    center = [int(x) for x in center]
    image_shape = image.GetSize()
    
    # patch half size
    phfx = dim[0] // 2
    phfy = dim[1] // 2
    phfz = dim[2] // 2

    x1 = center[0] - phfx
    x2 = center[0] + dim[0] - phfx
    if x1 < 0:
        x1 = 0
        x2 = dim[0]

    
    
    y1 = center[1] - phfy
    y2 = center[1] + dim[1] - phfy
    if y1 < 0:
        y1 = 0
        y2 = dim[1]

    z1 = center[2] - phfz
    z2 = center[2] + dim[2] - phfz
    if z1 < 0:
        z1 = 0
        z2 = dim[2]


    if x2 > image_shape[0]:
        x2 = image_shape[0]
        x1 = image_shape[0] - dim[0]
    
    if y2 > image_shape[1]:
        y2 = image_shape[1]
        y1 = image_shape[1] - dim[1]
        
    if z2 > image_shape[2]:
        z2 = image_shape[2]
        z1 = image_shape[2] - dim[2]
 
    return image[x1:x2, y1:y2, z1:z2]

    

def sitk_min_max_scale(image):
    min_max_filter.Execute(image)
    min_v = min_max_filter.GetMinimum()
    max_v = min_max_filter.GetMaximum()
    image = sitk.Cast(image, sitk.sitkFloat32)
    
    return (image - min_v )/ (max_v - min_v )


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

def register_image(fixed_image, moving_image,
                   transformParameterMap=None,
                   interpolator='nn',
                   dataset=None,
                   param_set=['translation', "affine"]):
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
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.SetFixedImage(fixed_image)
        elastixImageFilter.SetMovingImage(moving_image)
        
        if dataset is not None and 'copd' in str(dataset):
            # for COPD we'd like to use the cylinder mask
            elastixImageFilter.SetMovingMask(moving_image > 0)
            elastixImageFilter.SetFixedMask(fixed_image  > 0)
        
        # elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap(param_set))
        
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap(param_set[0]))
        elastixImageFilter.AddParameterMap(sitk.GetDefaultParameterMap(param_set[1]))
        
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
        return transformixImageFilter.GetResultImage(), transformParameterMap
        
        
def transform_points(moving, fixed, dataset, case):
    # propagate (transform) the points (landmarks) with the obtained transformation (deformation map)

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.LogToConsoleOff()
    # set moving image to initialize object correctly
    transformixImageFilter.SetMovingImage(moving)

    # register scans INVERSLY to transform points
    __, transformParameterMap = register_image(fixed_image=moving, moving_image=fixed, interpolator=None, dataset=dataset)
    
    # set previously obtained transformation parameters
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    
    # read key points csv file
    kps = pd.read_csv(dataset/'keypoints'/f'{case}.csv', header=None).values
    
    moving_kps = kps[:, [0, 1, 2]]
    fixed_kps = kps[:, [3, 4, 5]]
    
    with open(dataset/'moving_kpts.pts', 'w') as f:
        f.write('index\n')
        f.write(str(len(moving_kps))+'\n')
        for i, kp in enumerate(moving_kps):
            f.write(f'{kp[0]} {kp[1]} {kp[2]}\n')
    
    
    # set points to transform (inhale)
    transformixImageFilter.SetFixedPointSetFileName(str(dataset/'moving_kpts.pts'))
    transformixImageFilter.SetOutputDirectory(str(dataset))
    transformixImageFilter.Execute()
    
    moved_kps = parse_points_reg(str(dataset/'outputpoints.txt'))
    
    print(f'TRE before registration {compute_TRE(fixed_kps, moving_kps, moving.GetSpacing())}')
    print(f'TRE after registration {compute_TRE(fixed_kps, moved_kps, moving.GetSpacing())}')
    
    # save transformed points
    kps_path = (Path('processed_data')/dataset.name)/'keypoints/'
    kps_path.mkdir(parents=True, exist_ok=True)
    with open(kps_path/f'{case}.csv', 'w') as f:
        for i, kp in enumerate(moved_kps):
            f.write(f'{kp[0]},{kp[1]},{kp[2]},{fixed_kps[i][0]},{fixed_kps[i][1]},{fixed_kps[i][2]}\n')
    
def resample_volume(volume, interpolator = sitk.sitkLinear, new_spacing = [1, 1, 1]):
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())

def transform_resampled_kps(old, new, kp):
    """ Gets keypoint coordinates in the new image
        based on kp coordinates in the old image"""
    return new.TransformPhysicalPointToContinuousIndex(old.TransformContinuousIndexToPhysicalPoint(kp))
    
def resample_imgs_points(fixed_image, moving_img,
                         fixed_mask, moving_mask,
                         dataset, case):
    # for now resample only COPD
    if dataset is not None and 'copd' in str(dataset):
        
        if 'case_001' in str(case):
            # voxel_size = [0.63, 0.63, 3.16] # gets size (512, 512, 96)
            voxel_size = [1.44, 1.44, 3.14] # gets size (224, 224, 96)
        elif 'case_002' in str(case):
            # voxel_size = [0.65, 0.65, 2.65]
            voxel_size = [1.489, 1.489, 2.65]
        elif 'case_003' in str(case):
            # voxel_size = [0.65, 0.65, 3.29]
            voxel_size = [1.489, 1.489, 3.29]
        elif 'case_004' in str(case):
            # voxel_size = [0.59, 0.59, 3.29]
            voxel_size = [1.35, 1.35, 3.29]
        
        fixed_rsmp = resample_volume(fixed_image, new_spacing=voxel_size)
        moving_rsmp = resample_volume(moving_img, new_spacing=voxel_size)
        fixed_mask_rsmp = resample_volume(fixed_mask, new_spacing=voxel_size)
        moving_mask_rsmp = resample_volume(moving_mask, new_spacing=voxel_size)
        
        # transform points
        # read key points csv file
        kps = pd.read_csv((Path('processed_data')/dataset.name)/f'keypoints/{case}.csv', header=None).values
        
        moving_kps = kps[:, [0, 1, 2]]
        fixed_kps = kps[:, [3, 4, 5]]
        
        trsf_kps = np.zeros_like(kps)
        trsf_kps[:, [0, 1, 2]] = [transform_resampled_kps(moving_img, moving_rsmp, kp) for kp in moving_kps]
        trsf_kps[:, [3, 4, 5]] = [transform_resampled_kps(fixed_image, fixed_rsmp, kp) for kp in fixed_kps]
        
        # save the new kps
        with open((Path('processed_data')/dataset.name)/f'keypoints/{case}.csv', 'w') as f:
            for kp in trsf_kps:
                f.write(f'{kp[0]},{kp[1]},{kp[2]},{kp[3]},{kp[4]},{kp[5]}\n')
        
        return fixed_rsmp, moving_rsmp, fixed_mask_rsmp, moving_mask_rsmp
        
    else:
        return fixed_image, moving_img, fixed_mask, moving_mask

def main():
    for dataset in data_dir.iterdir():
        if not dataset.is_dir():
            continue
        if 'learn2reg' in str(dataset):
            continue
        cases = []
        with open(dataset/'train_cases.txt') as f:
            cases.extend(f.readlines())
        with open(dataset/'val_cases.txt') as f:
            cases.extend(f.readlines())
        cases = list(set([case.strip() for case in cases]))
        
        cases_path = (Path('processed_data')/dataset.name)/'scans/'
        cases_path.mkdir(parents=True, exist_ok=True)
        masks_path = (Path('processed_data')/dataset.name)/'lungMasks/'
        masks_path.mkdir(parents=True, exist_ok=True)
        
        for case in tqdm(cases):
            fixed_image = sitk.ReadImage(str(dataset/'scans'/f'{case}_insp.nii.gz'))
            moving_image = sitk.ReadImage(str(dataset/'scans'/f'{case}_exp.nii.gz'))
            fixed_mask = sitk.ReadImage(str(dataset/'lungMasks'/f'{case}_insp.nii.gz'))
            moving_mask = sitk.ReadImage(str(dataset/'lungMasks'/f'{case}_exp.nii.gz'))

            # register scan and mask with affine transformation
            moving_img_transformed, transformParameterMap = register_image(fixed_image, moving_image, interpolator=None, dataset=dataset)
            moving_mask_transformed, _ = register_image(fixed_mask, moving_mask, transformParameterMap=transformParameterMap, interpolator='nn')

                        
            # scale fixed image and moving image intensities to [0, 1]
            fixed_image_scaled = sitk_min_max_scale(fixed_image)
            moving_img_transformed_scaled = sitk_min_max_scale(moving_img_transformed)
            
            
            # TODO: Use inversion of the transformation to transform the points
            # instead of doing second reverse registration
            # transform keypoints coordinates after affine registration
            transform_points(moving_image, fixed_image, dataset, case)
            
            
            
            # ONLY AFTER POITNS TRANSFORMATION
            # resample images to the same voxel size
            # will resample images to the same dimensions if they were before the same
            
            
            fixed_img_rsmp, moving_img_rsmp, fixed_msk_rsml, moving_msk_rsml = resample_imgs_points(fixed_image_scaled, moving_img_transformed_scaled, fixed_mask, moving_mask_transformed, dataset, case)
            # fixed_img_rsmp, moving_img_rsmp, fixed_msk_rsml, moving_msk_rsml = fixed_image_scaled, moving_img_transformed_scaled, fixed_mask, moving_mask_transformed
            
            # save processed images
            sitk.WriteImage(fixed_img_rsmp, str(cases_path/f'{case}_insp.nii.gz'))
            sitk.WriteImage(moving_img_rsmp, str(cases_path/f'{case}_exp.nii.gz'))
            sitk.WriteImage(fixed_msk_rsml, str(masks_path/f'{case}_insp.nii.gz'),)
            sitk.WriteImage(moving_msk_rsml, str(masks_path/f'{case}_exp.nii.gz'))
            # break

if __name__ == '__main__':
    main()
