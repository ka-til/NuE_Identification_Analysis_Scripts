#Import necessary modules
from icecube import icetray, dataio, phys_services
from icecube import dataclasses
from icecube.icetray import I3Tray
from icecube.phys_services import I3Calculator
from icecube.hdfwriter import I3HDFWriter
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from glob import glob
import sys
import argparse
import numpy as np
import scipy as sp
import os

def select_beta(frame, geo):

    #print("calculate beta angle")

    mctree = frame["I3MCTree"]
    primary = mctree.primaries

    daughter = dataclasses.I3MCTree.first_child(mctree, primary[0].id)

    #pid value
    pid = frame['graphnet_dynedge_track_classification_track_pred'].value

    beta1_reco = frame['Beta1'].value
    median_angle = frame['median_angle'].value

    if beta1_reco >= 0 and beta1_reco <= 0.0002 and median_angle >= 0.62 and median_angle <= 0.77 and pid <= 0.2:
        
        return True

    return False

def create_dataset(infile, outfile, geo):
    #keys = ["graphnet_dynedge_energy_reconstruction_energy_pred", "Beta1", "median_angle", "daughter_type", "penergy", "senergy"]
    
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=infile)

    tray.AddModule(select_beta, 'vertex cut',
                   geo = geo,
                   Streams=[icetray.I3Frame.Physics])

    tray.AddModule('I3Writer','writer',
                   FileName=outfile,
                   DropOrphanStreams=[icetray.I3Frame.DAQ])

    tray.Execute()
    tray.Finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process IceCube data and create a dataset.")
    #parser.add_argument("--outdir", type=str, default='/data/user/akatil/electron_neutrino/for_real/dataset_complete/enriched_all_with_beta/', help="Give the output directory where files should be located")
    #parser.add_argument("--fold", type=str, default='48', help="Give the last three digits of the dataset")
    parser.add_argument("--dataset", type=str, default='120028', help="Give the last three digits of the dataset")
    parser.add_argument("--outdir", type=str, default='/data/user/akatil/electron_neutrino/for_real/dataset_complete/more_enriched_029/', help="Give the output directory where files should be located")

    args = parser.parse_args()

    outdir = args.outdir
    dataset = args.dataset
    #fold = args.fold

    #getting the geometry frame. We want this to calculate the mean of upgrade om x and y positions
    gcd_infile = dataio.I3File('/home/akatil/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2')

    f_geo = gcd_infile
    geo_frame = f_geo.pop_frame(icetray.I3Frame.Geometry)
    geo = geo_frame['I3Geometry']

    filelist = sorted(glob('/data/user/akatil/electron_neutrino/for_real/dataset_complete/enriched_029/'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
    #filelist = sorted(glob(f'/data/user/akatil/electron_neutrino/for_real/dataset_complete/enriched_all/folder_{fold}/upgrade_genie_level4_queso_*.i3.zst'))

    #print(filelist)

    #outloc = outdir+'/'+dataset+'/'#+'enriched_h5/'+dataset+'/'
    outloc = os.path.join(outdir, dataset, '')
    os.makedirs(outloc, exist_ok=True)

    for f in filelist:
        fname=f.split('/')[-1]
        #print (f"Processing file: {fname}")
        outfile=outloc+fname
        #outfile=outloc+fname.strip(".i3.zst")+".h5"
        #print([f])
        #print (f"Processing file: {outfile}")
        create_dataset([f], outfile, geo)

