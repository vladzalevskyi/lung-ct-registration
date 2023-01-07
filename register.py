import os
import sys; sys.path.insert(0, os.path.abspath("./"))

from pathlib import Path
import argparse
import numpy as np
import SimpleITK as sitk
from general_utils import utils
import time

this_path = Path().resolve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register a pair of lung CT images (inhale, exhale)')
    parser.add_argument("fixed", help="path to fixed image (nii.gz), inhale")
    parser.add_argument("moving", help="path to moving image (nii.gz), exhale")
    parser.add_argument("points_i", help="path to landmarks (.txt), inhale")
    # parser.add_argument("points_e", help="path to landmarks (.txt), exhale")
    args = parser.parse_args()

    start_time = time.time()
    test_path = this_path / 'test_data/'
    paramMaps_path = this_path / 'data/copd/paramMaps/'

    # read fixed image and create lung mask
    fixed_itk = sitk.ReadImage(str(test_path/args.fixed))
    fixed_lung_mask = utils.segment_lungs(image=fixed_itk,
                                            output_path=(test_path/(args.fixed.split('.')[0] + '_lm.nii.gz')))
    fixed_itk = utils.preprocess_image(fixed_itk)

    # read moving image and create lung mask
    moving_itk = sitk.ReadImage(str(test_path/args.moving))
    moving_lung_mask = utils.segment_lungs(image=moving_itk,
                                            output_path=(test_path/(args.moving.split('.')[0] + '_lm.nii.gz')))
    moving_itk = utils.preprocess_image(moving_itk)

    print(f'Lung masks ready in {time.time() - start_time}')

    # read exhale points
    points_inhale = np.loadtxt(test_path/args.points_i).astype(np.int16)
    # points_exhale = np.loadtxt(test_path/args.points_e).astype(np.int16)
    utils.save_pts_itk(points=points_inhale, path=(test_path/(args.points_i.split('.')[0] + '.pts')))

    print('Points read')

    # read params 11
    pm_affine = sitk.ReadParameterFile(str(paramMaps_path/'Parameters.Par0011.affine.txt'))
    pm_bspline_1 = sitk.ReadParameterFile(str(paramMaps_path/'Parameters.Par0011.bspline1_s.txt'))
    pm_bspline_2 = sitk.ReadParameterFile(str(paramMaps_path/'Parameters.Par0011.bspline2_s.txt'))

    # composed transformation

    print('Starting registration')
    start_reg = time.time()
    moving_reg, mov_param = utils.register_image_external(fixed_image=fixed_itk, fixed_mask=fixed_lung_mask, 
                                                    moving_image=moving_itk, moving_mask=moving_lung_mask, 
                                                    paramMaps=[pm_affine, pm_bspline_1, pm_bspline_2], print_console=False)
    print(f'Finished in: {time.time() - start_reg}')

    # transform lung mask                                                 
    moving_lm = utils.register_image_w_mask(fixed_image=None, moving_image=moving_lung_mask, fixed_mask=None, moving_mask=None,
                                            transformParameterMap=mov_param, interpolator='nn')[0]
    
    # export transformed images
    sitk.WriteImage(moving_reg, str(test_path/(args.moving.split('.')[0] + '_transformed.nii.gz')))
    sitk.WriteImage(moving_lm, str(test_path/(args.moving.split('.')[0] + '_lm_transformed.nii.gz')))

    # transform points for TRE computation
    points_inhale_moved = utils.transform_points_test(moving_itk=moving_itk, mov_param=mov_param, 
                                            fixed_points_path=(test_path/(args.points_i.split('.')[0] + '.pts')),
                                            output_dir=test_path)
    
    # print(f'TRE: {utils.compute_TRE(points_exhale, points_inhale_moved, voxel_spacing=moving_itk.GetSpacing())}')
    print(f'Run in {time.time()-start_time}')