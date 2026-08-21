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
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import iqr
import os

n_gr = dataclasses.I3Constants.n_ice_group
c = dataclasses.I3Constants.c

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

def beta_n(power, coeff, array):
    legendre = sp.special.eval_legendre(power, array)
    beta = coeff*sum(legendre)
    return beta 

def get_beta_values(ex, ey, ez):
    angle_arr = ([])
    for i in range(0, len(ex)):
        for j in range(i+1, len(ex)):
            if j <= len(ex):
                s = ex[i]*ex[j] + ey[i]*ey[j] + ez[i]*ez[j]
                s = s/(np.sqrt(ex[i]**2+ey[i]**2+ez[i]**2)*np.sqrt(ex[j]**2+ey[j]**2+ez[j]**2))
                angle_arr = np.append(angle_arr, s)

    N = len(angle_arr)
    
    if N > 1:
        coeff = 2/(N*(N-1))

        return beta_n(1, coeff, angle_arr), beta_n(2, coeff, angle_arr), beta_n(3, coeff, angle_arr), beta_n(4, coeff, angle_arr), beta_n(5, coeff, angle_arr)

    else:
        return [float('nan')] * 5

def get_cog(charge, x_arr, y_arr, z_arr):
    if len(charge) != 0:
        #weighted averae to get the CoG vertex
        total_charge = sum(charge)
        weighted_x = sum(x_arr*charge)/total_charge
        weighted_y = sum(y_arr*charge)/total_charge
        weighted_z = sum(z_arr*charge)/total_charge

    return weighted_x, weighted_y, weighted_z

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

def reco_time(frame, geo):

    pulses = frame["SplitInIcePulses_dynedge_v2_Pulses"]
    hits = pulses.apply(frame)
    #qTot = sum([hit.charge for entry in hits for hit in entry.data()])

    rx, ry, rz = frame['graphnet_dynedge_position_reconstruction_position_x_pred'].value, frame['graphnet_dynedge_position_reconstruction_position_y_pred'].value, frame['graphnet_dynedge_position_reconstruction_position_z_pred'].value

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
    frame["first_hit"] = dataclasses.I3Double(min_time)

    vertex_time = min_time - (n_gr/c)*d

    frame["reco_vertex_time"] = dataclasses.I3Double(vertex_time)

def get_var(frame, geo):

    #print("calculate beta angle")

    mctree = frame["I3MCTree"]
    primary = mctree.primaries

    daughter = dataclasses.I3MCTree.first_child(mctree, primary[0].id)

    #pid value
    pid = frame['graphnet_dynedge_track_classification_track_pred'].value

    #reconstructed position
    rx, ry, rz = frame['graphnet_dynedge_position_reconstruction_position_x_pred'].value, frame['graphnet_dynedge_position_reconstruction_position_y_pred'].value, frame['graphnet_dynedge_position_reconstruction_position_z_pred'].value

    #reconstructed time
    reco_vertex_time = frame["reco_vertex_time"].value
    
    reco_energy = frame["graphnet_dynedge_energy_reconstruction_energy_pred"].value

    #reconstructed zenith
    reco_dir_x = frame['graphnet_dynedge_direction_reconstruction_dir_x_pred'].value
    reco_dir_y = frame['graphnet_dynedge_direction_reconstruction_dir_y_pred'].value
    reco_dir_z = frame['graphnet_dynedge_direction_reconstruction_dir_z_pred'].value

    #All the pulses in the event
    pulses = frame["SplitInIcePulses_dynedge_v2_Pulses"]
    hits = pulses.apply(frame)
    
    ex, ey, ez = -reco_dir_x, -reco_dir_y, -reco_dir_z

    daughter.pos = dataclasses.I3Position(rx, ry, rz)
    daughter.time = reco_vertex_time 

    #change tau shape from starting cascade to 
    if daughter.type == 15 or daughter.type == -15:
        daughter.shape = dataclasses.I3Particle.Cascade
     
    om_prop = []
    for entry in hits:
        if geo.omgeo.get(entry.key()).omtype.name == 'mDOM':
            #print('hit length', len(entry.data())) 
            #position on the om
            omgeo_pos = geo.omgeo.get(entry.key()).position
            ox, oy, oz = omgeo_pos

            #Get Angular Distribution
            #vector b/w particle and OM
            hx = daughter.pos.x - ox
            hy = daughter.pos.y - oy
            hz = daughter.pos.z - oz

            #dot product
            s = ex*hx + ey*hy + ez*hz

            angle = -s/(np.sqrt(hx**2+hy**2+hz**2))#np.arccos(-s/(np.sqrt(hx**2+hy**2+hz**2)))

            omgeo_pos = geo.omgeo.get(entry.key()).position
            
            d = np.sqrt((daughter.pos.x-ox)**2+(daughter.pos.y-oy)**2+(daughter.pos.z-oz)**2)

            for hit in entry.data():

                #Get Time Residuals
                time_residual = I3Calculator.time_residual(daughter, omgeo_pos,
                                                                 hit.time)
                '''
                for each hit appeding the following
                om positions
                time
                charge
                distance between vertex om
                time residual in nanoseconds
                angle between the particle and OM in radians
                vector b/w particle and OM for beta14 calculations
                '''    
                om_prop.append((ox, oy, oz, hit.charge, d, time_residual, angle, hx, hy, hz))

    if len(om_prop) != 0:

        x_arr, y_arr, z_arr, charge_arr, dist_arr, tres_arr, angle_arr, hx_arr, hy_arr, hz_arr = map(np.array, zip(*om_prop))
    
        #beta1_reco, beta2_reco, beta3_reco, beta4_reco, beta5_reco = get_beta_values(hx_arr, hy_arr, hz_arr)
        
        #median values
        median_angle = np.median(angle_arr)
        median_tres = np.median(tres_arr)
        median_distance = np.median(dist_arr)

        #iqr values
        iqr_angle = iqr(angle_arr)
        iqr_tres = iqr(tres_arr)
        iqr_distance = iqr(dist_arr)

        #skew values
        skew_angle = skew(angle_arr)
        skew_tres = skew(tres_arr)
        skew_distance = skew(dist_arr)

        #skew values
        kurtosis_angle = kurtosis(angle_arr)
        kurtosis_tres = kurtosis(tres_arr)
        kurtosis_distance = kurtosis(dist_arr)
        
        #cog_x, cog_y, cog_z = get_cog(charge_arr, x_arr, y_arr, z_arr)
        #cog_distance = np.sqrt((rx-cog_x)**2+(ry-cog_y)**2+(rz-cog_z)**2)

        total_charge = sum(charge_arr)

        '''
        Skipping the key features from the first iteration
        event_features = {
            "Beta1Reco": beta1_reco,
            "Beta2Reco": beta2_reco,
            "Beta3Reco": beta3_reco,
            "Beta4Reco": beta4_reco,
            "Beta5Reco": beta5_reco,
            "MedianAngle": median_angle,
            "MediantRes": median_tres,
            "MedianDistance": median_distance,
            "CogDistance": cog_distance,
        }
        '''

        if 'MediantRes' in frame:
            del frame['MediantRes']
        
        event_features = {
            "IqrAngle": iqr_angle,
            "IqrtRes": iqr_tres,
            "IqrDistance": iqr_distance,
            "SkewAngle": skew_angle,
            "SkewtRes": skew_tres,
            "SkewDistance": skew_distance,
            "KurtosisAngle": kurtosis_angle,
            "KurtosistRes": kurtosis_tres,
            "KurtosisDistance": kurtosis_distance,
            "TotalCharge": total_charge,
            "MediantRes": median_tres
        }

    
        for key, value in event_features.items():
            frame[key] = dataclasses.I3Double(value)

        return True

    return False

def create_dataset(infile, outfile, mean_ux, mean_uy, mean_uz, geo):
    #keys = ["graphnet_dynedge_energy_reconstruction_energy_pred", "Beta1", "median_angle", "daughter_type", "penergy", "senergy"]
    
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=infile)

    
    '''
    tray.AddModule(upgrade_events, 'cuts',
                   mean_x = mean_ux,
                   mean_y = mean_uy,
                   mean_z = mean_uz,
                   Streams=[icetray.I3Frame.Physics]) #For the second iteration I already selected events that are in the upgrade and calculated the reco time

    tray.AddModule(reco_time, 'process',
                   geo = geo,
                   Streams=[icetray.I3Frame.Physics])
    '''

    tray.AddModule(get_var, 'necessary vars',
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
    parser.add_argument("--dataset", type=str, default='120029', help="Give the last three digits of the dataset")
    #parser.add_argument("--outdir", type=str, default='/data/user/akatil/electron_neutrino/for_real/dataset_complete/enriched_029_with_all_var/', help="Give the output directory where files should be located")
    parser.add_argument("--outdir", type=str, default='/data/user/akatil/electron_neutrino/for_real/dataset_complete/enriched_029_median_iqr_skew_kurtosis_beta/', help="Give the output directory where files should be located")

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
    
    filelist = sorted(glob(f'/data/user/akatil/electron_neutrino/for_real/dataset_complete/enriched_029_with_all_var/{dataset}/upgrade_genie_level4_queso_*.i3.zst'))
    #filelist = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01_with_pos_reco/'+dataset+'/upgrade_genie_level4_queso_*.i3.zst')) First Iter
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