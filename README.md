### Create a virtual miniconda environment
```
```

### To download the data:

1. Run 
```
pip install -r requirements.txt
```
to have gdown installed (tool to download from GDrive

2. Go to data folder
```
cd data/
```
3. Run
```
gdown --folder 1yHWLQEK9c1xzggkCC4VX0X4To7BBDqu5 # for EMPIRE Data

curl https://zenodo.org/record/3835682/files/training.zip?download=1 --output learn2reg.zip # for Learn2Reg Data

gdown --folder 1cARJcCKWtGP44p3e0X4Umpf_ISB-64lC # for our 4 COPD train cases
```

### TODO:

1. Write a script to load and process data to have it in the format

```
.
└── data
    └── learn2reg
        ├── keypoints
│       │   ├── case_001.csv
│       │   ├── case_002.csv
│       │   ├── ..........
        ├── lungMasks
│       │   ├── case_001_exp.nii.gz
│       │   ├── case_001_insp.nii.gz
│       │   ├── case_002_exp.nii.gz
│       │   ├── case_002_insp.nii.gz
│       │   ├── ..........
        └── scans
│       │   ├── case_001_exp.nii.gz
│       │   ├── case_001_insp.nii.gz
│       │   ├── case_002_exp.nii.gz
│       │   ├── case_002_insp.nii.gz
│       │   ├── ..........
```
### After the data is in the right format run the script to fix train/val splits
```
$ python data/fix_train_partitions.py

```

### Test TODO
1. Make sure you have the envioronment copy set up
```
conda create --name myenv --file env-spec-file.txt
```
2. You will need to have a directory called ```paramMaps``` in the location ```data/copd/paramMaps```. Three files must be there:
- Parameters.Par0011.affine.txt
- Parameters.Par0011.bspline1_s.txt
- Parameters.Par0011.bspline2_s.txt

3. You will need to save your Dir-Lab COPDgene **test** images in the test_data folder. These will need to be:

- inhale image (nii.gz)
- exhale image (nii.gz)
- inhale landarmks (.txt)

4. From the repository directory, run the file ```register.py```:
```
python register.py fixed_image_path moving_image_path fixed_image_points_path
```
5. The resulting files will be saved in the ```test_data``` folder:
- inhale lung mask (_lm.nii.gz)
- exhale lung mask (_lm.nii.gz)
- exhale lung mask transformed (_lm_transformed.nii.gz)
- transformed inhale points (transformed_points.txt)
