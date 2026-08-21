import numpy as np
from glob import glob
from icecube.icetray import I3Tray
from icecube import dataclasses, dataio, icetray, simclasses
import sys
import os
import argparse

#frame_count = 0

def frame_counter(frame):
    '''
    This function adds unique ids to each frame.

    Arguments
    frame: DAQ or Physics frames
    '''
    global frame_count #use the global variable
    frame_count = frame_count+1
    #print(frame_count)
    frame['Counter'] = dataclasses.I3Double(frame_count)
    return True

def create_dataset(infile, outfile):
    '''
    This function creates the final dataset. It skips some keys not used for further analysis

    Arguments
    infile: Input file 
    outfile: Output file
    '''
    tray = I3Tray()
    #print(infile)
    tray.AddModule('I3Reader', 'reader', FilenameList=infile)

    #metadata_keys += config['MCmetadata_keys']

    #run the frame counter function
    tray.AddModule(frame_counter, 'counter',
                   Streams=[icetray.I3Frame.Physics])

    tray.AddModule('I3Writer','writer',
           FileName=outfile)

    tray.Execute()
    tray.Finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple time reco.")
    parser.add_argument("--dataset", type=str, default='CC', help="CC or NC or muon_neutrino")

    args = parser.parse_args()

    dataset = args.dataset

    
    #modify this step if files for NuE analysis are not used.
    if dataset == 'CC':
        frame_count = 1e10
        path = '/data/user/akatil/electron_neutrino/for_real/dataset_complete/batches/CC/'
        outloc = '/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction/CC_time/'

    if dataset == 'NC':
        frame_count = 2e10
        path = '/data/user/akatil/electron_neutrino/for_real/dataset_complete/batches/NC/'
        outloc = '/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction/NC_time/'

    if dataset == 'muon_neutrino':
        frame_count = 3e10
        path = '/data/user/akatil/electron_neutrino/for_real/dataset_complete/batches/muon_neutrino/'
        outloc = '/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction/muon_neutrino_time/'

    #The files are split into groups of 100s so that step 3 runs quickly. Run the bash script before running this loop. 
    #bash script in '/data/user/akatil/electron_neutrino/for_real/dataset_complete/batches'
    #loop through each file from the file list and create a new output file after making necessary cuts.
    for root, _, files in os.walk(path):
        filelist = sorted(glob(os.path.join(root, '*i3.zst')))
        print(root)
        for f in filelist:
            fname=f.split('/')[-1]
            #print (f"Processing file: {fname}")         
            outfile=outloc+fname
            #print([f])
            #print (f"Processing file: {outfile}")
            create_dataset([f], outfile)
