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

def get_mean_upgrade_positions(geo):
    '''
    Get the mean x, y and z positions of upgrade optical module

    Arguments
    geo: Geometry frame

    Returns
    mean x, y, z of upgrade doms
    '''
    ux, uy, uz = [], [], []

    #get the positions of upgrade DOMs
    for omkey in geo.omgeo.keys():
        oKey = geo.omgeo.get(omkey)
        domPos = oKey.position
        if domPos.z > 1000: #Don't want IceTop in the plot
            continue
        if omkey[0] > 86: #upgrade
            if domPos.z <= -150 and domPos.z >= -500: #want to focus on the concentrated region of upgrade in the clear ice.
                ux.append(domPos.x)
                uy.append(domPos.y)
                uz.append(domPos.z)

    #get the mean positions
    mean_x = np.mean(ux)
    mean_y = np.mean(uy)
    mean_z = np.mean(uz)

    return mean_x, mean_y, mean_z

def upgrade_events(frame, mean_x, mean_y, mean_z):

    #print("print cascade_upgrade_events")

    mctree = frame['I3MCTree']
    primary = mctree.primaries
        
    px, py, pz = frame['graphnet_dynedge_position_reconstruction_position_x_pred'].value, frame['graphnet_dynedge_position_reconstruction_position_y_pred'].value, frame['graphnet_dynedge_position_reconstruction_position_z_pred'].value #primary[0].pos #change everything to reco
    
    daughter = dataclasses.I3MCTree.first_child(mctree, primary[0].id)

    reco_energy = frame["graphnet_dynedge_energy_reconstruction_energy_pred"].value

    #if primary[0].energy <= 100:
    if reco_energy <= 100:
        if pz <= mean_z+150 and pz >= mean_z-150: #previously no cut in z
            if px <= mean_x+50 and px >= mean_x-50:
                if py <= mean_y+40 and py >= mean_y-40:
                    #if daughter.type == 11 or daughter.type == -11 or daughter.type == 15 or daughter.type == -15 or daughter.type == 12 or daughter.type == -12 or daughter.type == 14 or daughter.type == -14 or daughter.type == 16 or daughter.type == -16:
                    return True

    return False

def create_dataset(infile, outfile, mean_ux, mean_uy, mean_uz, geo):
    #keys = ["graphnet_dynedge_energy_reconstruction_energy_pred", "Beta1", "median_angle", "daughter_type", "penergy", "senergy"]
    
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=infile)

    tray.AddModule(upgrade_events, 'cuts',
                   mean_x = mean_ux,
                   mean_y = mean_uy,
                   mean_z = mean_uz,
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
    parser.add_argument("--outdir", type=str, default='/data/user/akatil/electron_neutrino/for_real/dataset_complete/just_upgrade_029/', help="Give the output directory where files should be located")

    args = parser.parse_args()

    outdir = args.outdir
    dataset = args.dataset
    #fold = args.fold

    #getting the geometry frame. We want this to calculate the mean of upgrade om x and y positions
    gcd_infile = dataio.I3File('/home/akatil/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2')

    f_geo = gcd_infile
    geo_frame = f_geo.pop_frame(icetray.I3Frame.Geometry)
    geo = geo_frame['I3Geometry']

    ux, uy, uz = get_mean_upgrade_positions(geo)

    filelist = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01_with_pos_reco/'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
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
        create_dataset([f], outfile, ux, uy, uz, geo)
