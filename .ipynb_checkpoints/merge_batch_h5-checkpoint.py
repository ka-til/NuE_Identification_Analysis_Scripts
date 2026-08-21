'''
Script based on code from Sourav
'''

import os
import h5py
import argparse
import numpy as np
from glob import glob
import random

parser = argparse.ArgumentParser(description="Merge h5 files")          
parser.add_argument("--dataset", type=str, default='CC', help="Indicate whether to use CC or NC dataset.")
parser.add_argument("--outdir", type=str, default='/data/user/akatil/electron_neutrino/for_real/dataset_complete/merger/', help="Give the output directory where files should be located")

args = parser.parse_args() 

dataset = args.dataset
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

#loop through batches to get all the h5 files

file_list = []

folder = 0

base_folder = '/data/user/akatil/electron_neutrino/for_real/dataset_complete/variables/'
for root, _, files in os.walk(base_folder+dataset+'/'):
    folder += 1
    print(f'folder number is {folder}')
    f = sorted(glob(os.path.join(root, '*.h5')))
    file_list = file_list+f #get the list of .h5 files that need to be merged
   
    #if folder == 3:
       #break

random.seed(42)
random.shuffle(file_list)

if dataset == 'CC':
    batch_size = 5000
elif dataset == 'NC':
    batch_size = 15000
elif dataset == 'muon_neutrino':
    batch_size = 10000
else:
    raise ValueError(f"Unsupported dataset type: {dataset}. Expected 'CC' or 'NC'.")

num_batches = (len(file_list) + batch_size - 1)//batch_size

for batch_id in range(num_batches):
    batch_files = file_list[batch_id*batch_size:(batch_id+1)*batch_size]

    dimensions = {} #Initialize an empty dictionary to store column dimensions
    dtypes = {} #Initialize an empty dictionary to store column data types

    for infile in batch_files:
        #Open h5 files for reading
        with h5py.File(infile, 'r') as f:
            #iterate over keys
            for key in f.keys():
                #skip if the key is '__I3Index__' #Contains starting and stopping position
                #if individual events. could have used this if the event ids were unique.
                if key == '__I3Index__':
                    continue

                #If key not in dimensions, add with count 0
                if key not in dimensions:
                    dimensions[key] = 0
                    #data types of the key are stored in dtypes dictionary
                    dtypes[key] = f[key].dtype
            
                #Based on the row count increase the dimensions
                dimensions[key] += f[key].shape[0]


    print(len(file_list))
    print(dimensions)

    #initialize a tracker with key and 0 shape
    row_tracker = dict((key, 0) for key in dimensions)

    #merge the h5 files into a single file

    outfilename = os.path.join(outdir, f'merged_{dataset}_{batch_id}.h5')#outdir+f'merged_{dataset}.h5'

    with h5py.File(outfilename, 'w') as outfile:
        #Create a dataset for filename with variable length byte strings
        outfile.create_dataset('filename', (dimensions['PrimaryEnergy'],), dtype=h5py.special_dtype(vlen=bytes))
     
        #Create datasets for the keys in the file
        for key in dimensions:
            outfile.create_dataset(key, (dimensions[key],), dtype=dtypes[key])

        #Process each input file for merging
        for infile in batch_files:
            with h5py.File(infile, 'r') as f:
                if len(f.keys()) <= 1:
                    continue


                #Get the number of events in the current input file.
                n_events = f['PrimaryEnergy'].shape[0]


                #Get filename start and end into into the output filename dataset
                outfile['filename'][row_tracker['PrimaryEnergy']: row_tracker['PrimaryEnergy']+n_events] = bytes(infile, encoding='ASCII')


                #Copy data from other columns
                for key in dimensions:
                    size = f[key].shape[0]
                    outfile[key][row_tracker[key]:row_tracker[key]+size] = f[key]
                    row_tracker[key] += size


    outfile.close()    
