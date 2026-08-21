#simple code for time reconstruction since there was an issue training GraphNeT time reconstruction

from icecube import dataclasses, dataio, icetray, simclasses
import numpy as np
from glob import glob
from icecube.icetray import I3Tray
import sys
import argparse

#Getting the required constants
n_gr = dataclasses.I3Constants.n_ice_group #for speed of photon use the group refractive index
c = dataclasses.I3Constants.c

def reco_time(frame, geo):
    '''
    This function calculates the reconstructed time using the reconstructed vertex and the om position of the first hit time.

    This function creates the final dataset. It skips some keys not used for further analysis

    Arguments
    frame: DAQ or Physics frames
    geo: Geometry frame

    Adds first hit time and reconstructed time to the I3Frame
    
    '''

    pulses = frame["SplitInIcePulses_dynedge_v2_Pulses"]
    hits = pulses.apply(frame)
    #qTot = sum([hit.charge for entry in hits for hit in entry.data()])

    rx, ry, rz = frame['graphnet_dynedge_position_reconstruction_position_x_pred'].value, frame['graphnet_dynedge_position_reconstruction_position_y_pred'].value, frame['graphnet_dynedge_position_reconstruction_position_z_pred'].value

    #run a loop to find the minimum
    min_time = 1e9
    #hit_charge = 0
    for entry in hits:
        for hit in entry.data():
            #hit_charge += hit.charge
            if hit.time < min_time:
                min_time = hit.time
                ox, oy, oz = geo.omgeo.get(entry.key()).position
                d = np.sqrt((ox-rx)**2+(oy-ry)**2+(oz-rz)**2)

    #print(min_time)
    #saving the first hit to I3Frame
    frame["first_hit"] = dataclasses.I3Double(min_time)
    
    #Calculate the time vertex time given the minimum time
    vertex_time = min_time - (n_gr/c)*d

    frame["reco_vertex_time"] = dataclasses.I3Double(vertex_time)

def create_dataset(infile, outfile, geo):
    '''
    This function creates the final dataset. It skips some keys not used for further analysis

    Arguments
    infile: Input file 
    outfile: Output file

    geo: Geometry frame
    '''
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=infile)

    #metadata_keys += config['MCmetadata_keys']

    tray.AddModule(reco_time, 'process',
                   geo = geo,
                   Streams=[icetray.I3Frame.Physics])

    tray.AddModule('I3Writer','writer',
           FileName=outfile)

    tray.Execute()
    tray.Finish()

#This only runs when the file is directly run and not when it is imported
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple time reco.")
    parser.add_argument("--dataset", type=str, default='CC', help="CC or NC or muon_neutrino") #The files used here must pass step 1 and step 2. 

    args = parser.parse_args()

    dataset = args.dataset

    #getting the geometry frame.
    gcd_infile = dataio.I3File('/home/akatil/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2')

    f_geo = gcd_infile
    geo_frame = f_geo.pop_frame(icetray.I3Frame.Geometry)
    geo = geo_frame['I3Geometry']

    #modify this if using this step if files for NuE analysis are not used.
    if dataset == 'CC':

        filelist = sorted(glob('/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction/CC/upgrade_genie_level4_queso_*.i3.zst'))

        outloc = '/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction/CC_time/'

    if dataset == 'NC':

        filelist = sorted(glob('/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction/NC/upgrade_genie_level4_queso_*.i3.zst'))

        outloc = '/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction/NC_time/'

    if dataset == 'muon_neutrino':

        filelist = sorted(glob('/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction/muon_neutrino/upgrade_genie_level4_queso_*.i3.zst'))

        outloc = '/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction/muon_neutrino_time/'

    #loop through each file from the file list and create a new output file after making necessary cuts.
    for f in filelist:
        fname=f.split('/')[-1]
        #print (f"Processing file: {fname}")
        outfile=outloc+fname
        #print([f])
        #print (f"Processing file: {outfile}")
        create_dataset([f], outfile, geo)
