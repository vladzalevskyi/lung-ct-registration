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
```$ python data/fix_train_partitions.py```s


### To run SynthMorph predictions for COPD cases:
1. Preprocess the data by ensuring directory structure as described above

2. Run the following script to preprocess the data
```$ python data/preprocess_data.py```

3. For each image resolution rebuild the model using
```$ python synthmorph/convert_tf2torch.py```  
and chaning inside the file the resolution to the input image. For now the COPD is resized to (512, 512, 96)
4. Run ```$ python synthmorph/predict_flow.py``` to get the deformation field matrix for each case
5. Run ```$ python synthmorph/transform_kps.py``` to transform keypoints to the new space for each case and calculate resulting TRE.

### To activate conda use

eval "$(/home/mira1/miniconda3/bin/conda shell.bash hook)"