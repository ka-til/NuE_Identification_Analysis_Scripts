#Import necessary modules
from icecube import icetray, dataio, phys_services
from icecube import dataclasses
from icecube.icetray import I3Tray
from glob import glob
import sys
import argparse
import numpy as np

def get_var(frame, dom_type='mDOM'):

    '''
    This function does not use any reconstructed variables

    pass the frame to the function.
    the function adds the necessary new keys to the frame object

    1) Total charge in IC92, mDOM and DEgg
    2) Time Residual
    3) Distance
    4) Angle
    4) Vector b/W OM and particle
    5) Center of Gravity of hits
    6) Time
    '''
    
    mctree = frame["I3MCTree"]
    primary = mctree.primaries

    daughter = dataclasses.I3MCTree.first_child(mctree, primary[0].id)
    
    px, py, pz = primary[0].pos
    
    pulses = frame["SplitInIcePulses_dynedge_v2_Pulses"]
    hits = pulses.apply(frame)
    qTot = sum([hit.charge for entry in hits for hit in entry.data()])

    hits_mDOM = frame["SplitInIcePulses_dynedge_v2_Pulses_mDOMs_Only"] #mDOM pulses only
    qTot_mDOM = sum([hit.charge for entry in hits_mDOM for hit in entry.data()])

    hits_dEgg = frame["SplitInIcePulses_dynedge_v2_Pulses_dEggs_Only"] #dEgg pulses only
    qTot_dEgg = sum([hit.charge for entry in hits_dEgg for hit in entry.data()])

    frame['TotalChargeIC92'] = qTot
    frame['TotalCharge_mDOM'] = qTot_mDOM
    frame['TotalCharge_dEgg'] = qTot_dEgg

    val_Map = dataclasses.I3MapKeyVectorDouble()
    
    x_arr, y_arr, z_arr, charge, time = ([]), ([]), ([]), ([]), ([])
    for entry in hits:
        if geo.omgeo.get(entry.key()).omtype.name == 'dom_type':
            hit_count = 0
            for hit in entry.data():

                ######## time residual #########
                diff = I3Calculator.time_residual(daughter, geo.omgeo.get(entry.key()).position,
                                                   hit.time)
                
                ###### distance #########
                wx, wy, wz = daughter.pos
                x, y, z = geo.omgeo.get(entry.key()).position
                d = np.sqrt((wx-x)**2+(wy-y)**2+(wz-z)**2)
                
                ##### angle distributions #####
                theta  = daughter.dir.zenith
                phi = daughter.dir.azimuth

                #unit vector for particle direction
                e_x = -np.sin(theta)*np.cos(phi) 
                e_y = -np.sin(theta)*np.sin(phi)
                e_z = -np.cos(theta)
                
                #vector b/w particle and OM
                h_x = wx - x 
                h_y = wy - y 
                h_z = wz - z 

                #dot product
                s = e_x*h_x + e_y*h_y + e_z*h_z

                angle = np.arccos(-s/(np.sqrt(h_x**2+h_y**2+h_z**2)))
                
                ####### Get values required to calculat CoG ###########
                x_arr = np.append(x_arr, x)
                y_arr = np.append(y_arr, y)
                z_arr = np.append(z_arr, z)
                charge = np.append(charge, hit.charge)
                time = np.append(time, hit.time)
                val_arr = np.array[hit_count, diff, d, angle, h_x, h_y, h_z] #saving time residual, distance, angle, vector b/w particle and OM
                
                hit_count += 1
            val_Map.update({entry.key(): dataclasses.I3VectorDouble(biGauss_values)})
                
    if len(charge) != 0:
        #weighted averae to get the CoG vertex
        weighted_x = sum(x_arr*charge)/sum(charge)
        weighted_y = sum(y_arr*charge)/sum(charge)
        weighted_z = sum(z_arr*charge)/sum(charge)
        mtime = min(time)

        cog_position = dataclasses.I3Position(weighted_x, weighted_y, weighted_z)
        min_time = dataclasses.I3Double(mtime)

        frame["cog_position"] = cog_position
        frame["first_time"] = min_time

def make_step2(infile, outfile, interaction_type, mean_ux, mean_uy, mean_uz):
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=infile)

    tray.AddModule(get_var, 'get values associalted with mc truth',
                   dom_type = 'mDOM',
                   Streams=[icetray.I3Frame.Physics])
    
    tray.AddModule('I3Writer','writer',
           FileName=outfile)

    tray.Execute()
    tray.Finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process IceCube data and create a dataset.")
    parser.add_argument("--interaction", type=int, default=1, help="Interaction type, 1 is CC electron neutrino and 0 is NC Events")
    parser.add_argument("--dataset", type=str, default='028', help="Give the last three digits of the dataset")
    parser.add_argument("--outdir", type=str, default='/data/user/akatil/electron_neutrino/for_real/dataset_complete/', help="Give the output directory where files should be located")

    args = parser.parse_args()

    interaction_type = args.interaction
    dataset = args.dataset
    outdir = args.outdir

    #getting the geometry frame. We want this to calculate the mean of upgrade om x and y positions
    gcd_infile = dataio.I3File('/home/akatil/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2')

    f_geo = gcd_infile
    geo_frame = f_geo.pop_frame(icetray.I3Frame.Geometry)
    geo = geo_frame['I3Geometry']

    ux, uy, uz = get_mean_upgrade_positions(geo)

    if interaction_type == 1:

        filelist1 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/120'+dataset+'upgrade_genie_level4_queso_120028_000*.i3.zst'))
        #NuE
        filelist2 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/121'+dataset+'/upgrade_genie_level4_queso_121028_000*.i3.zst'))
        #NuEBar

        filelist = filelist1 + filelist2
        outloc = outdir+'CC/'

    if interaction_type == 0:

        filelist1 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/120'+dataset+'/upgrade_genie_level4_queso_120028_000*.i3.zst'))
        filelist2 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/121'+dataset+'/upgrade_genie_level4_queso_121028_000*.i3.zst'))
        filelist3 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/140'+dataset+'/upgrade_genie_level4_queso_140028_000*.i3.zst'))
        filelist4 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/141'+dataset+'/upgrade_genie_level4_queso_141028_000*.i3.zst'))
        filelist5 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/160'+dataset+'/upgrade_genie_level4_queso_160028_000*.i3.zst'))
        filelist6 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/161'+dataset+'/upgrade_genie_level4_queso_161028_000*.i3.zst'))

        filelist = filelist1+filelist2+filelist3+filelist4+filelist5+filelist6
        outloc = outdir+'NC/'

    for f in filelist[0:10]:
        fname=f.split('/')[-1]
        #print (f"Processing file: {fname}")
        outfile=outloc+fname
        #print([f])
        #print (f"Processing file: {outfile}")
        create_dataset([f], outfile, interaction_type, ux, uy, uz)















































                