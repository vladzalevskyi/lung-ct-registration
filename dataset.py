import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List

import nibabel as nib
import csv


# TODO: Add MIND images/features???
# TODO: specific data preprocessing for different datasets 

class LungDatasets(Dataset):
    def __init__(self, sources, root_dir:Path,
                 partitions: List[str] = ['train', 'val'],
                 transform=None):
        self.sources = sources
        self.root_dir = root_dir
        self.partitions = partitions
        self.transform = transform
        
        
        self.get_image_paths()

    def __len__(self):
        total_images = 0
        for part in self.partitions:
            for source in self.sources:
                split_path = self.root_dir/f'{source}/{part}_cases.txt'
                if not split_path.exists():
                    raise ValueError(f'Partition {part} not found for {source} in {self.root_dir}')
                with open(split_path, 'r') as f:
                    cases = f.readlines()
                    total_images += len(cases)
        return total_images
    
    
    def __getitem__(self, idx):
        
        # retrieve some metadata
        sample = {'idx':idx}
        dataset = self.image_paths['scan_i'][idx].parent.parent.name
        case =  self.image_paths['scan_i'][idx].name[:8]
        sample['dataset'] = dataset
        sample['case'] = case
        
        
        if dataset == 'learn2reg':
            # load images
            sample['scan_i'] = torch.from_numpy(nib.load(self.image_paths['scan_i'][idx]).get_fdata()).float()
            sample['scan_e'] = torch.from_numpy(nib.load(self.image_paths['scan_e'][idx]).get_fdata()).float()
            sample['mask_i'] = torch.from_numpy(nib.load(self.image_paths['mask_i'][idx]).get_fdata()).float()
            sample['mask_e'] = torch.from_numpy(nib.load(self.image_paths['mask_e'][idx]).get_fdata()).float()
            
            # load keypoints
            corrfield = torch.empty(0,6)
            with open(self.image_paths['kps'][idx], newline='') as csvfile:
                fread = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in fread:
                    corrfield = torch.cat((corrfield,torch.from_numpy(np.array(row).astype('float32')).float().view(1,6)),0)

        
        # keypoints format same across dataset so just reorder
        sample['kps_e'] = torch.stack((corrfield[:,2]/207*2-1,
                                       corrfield[:,1]/191*2-1,
                                       corrfield[:,0]/191-2-1),1)
        
        sample['kps_i'] = torch.stack((corrfield[:,5]/207*2-1,
                                       corrfield[:,4]/191*2-1,
                                       corrfield[:,3]/191-2-1),1)     
        
        return sample
    
    
    def get_image_paths(self):
        self.image_paths = {'scan_i':[], 'scan_e':[],
                            'mask_i':[], 'mask_e':[],
                            'kps':[]}
        
        for part in self.partitions:
            for source in self.sources:
                split_path = self.root_dir/f'{source}/{part}_cases.txt'
                
                if not split_path.exists():
                    raise ValueError(f'Partition {part} not found for {source} in {self.root_dir}')
                
                with open(split_path, 'r') as f:
                    
                    cases = [x.strip() for x in f.readlines()]
                    
                    for case in cases:
                        self.image_paths['scan_i'].append(self.root_dir/f'{source}/scans/{case}_insp.nii.gz')
                        self.image_paths['scan_e'].append(self.root_dir/f'{source}/scans/{case}_exp.nii.gz')
                        self.image_paths['mask_i'].append(self.root_dir/f'{source}/lungMasks/{case}_insp.nii.gz')
                        self.image_paths['mask_e'].append(self.root_dir/f'{source}/lungMasks/{case}_exp.nii.gz')
                        self.image_paths['kps'].append(self.root_dir/f'{source}/keypoints/{case}.csv')
        
        # check that all paths exist
        for k in self.image_paths.keys():
            for p in self.image_paths[k]:
                try:
                    assert p.exists()
                except AssertionError:
                    raise ValueError(f'Path {p} does not exist')