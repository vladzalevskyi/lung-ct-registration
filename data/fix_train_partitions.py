
from pathlib import Path
import argparse

import random

parser = argparse.ArgumentParser(description='Splits image sets into train and validation.')
parser.add_argument('-s', '--split', dest='split', type=float, default=0.2, help='Split ratio for validation set (default: 0.2)') 
args = parser.parse_args()


if __name__ == '__main__':
    print('Splitting data with a valiation proportion of ', args.split)
    for path in Path('data').iterdir():
        if path.is_dir():
            cases = list(set(str(x.name)[:8] for x in (path/'scans').iterdir()))
            random.shuffle(cases)
            train_cases = cases[:int(len(cases)*(1-args.split))]
            validation_cases = cases[int(len(cases)*(1-args.split)):]
            
            # fix train cases
            with open(path/'train_cases.txt', 'w') as f:
                for case in train_cases:
                    f.write(case + '\n')
            
            # fix validation cases
            with open(path/'val_cases.txt', 'w') as f:
                for case in validation_cases:
                    f.write(case + '\n')
