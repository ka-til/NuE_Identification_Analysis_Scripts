from icecube import icetray, dataio, phys_services
from icecube import dataclasses
from icecube.icetray import I3Tray
from icecube.phys_services import I3Calculator
from icecube.hdfwriter import I3HDFWriter
from icecube.tableio import I3TableWriter
from icecube.hdfwriter import I3HDFTableService
from glob import glob
import sys
import os
import argparse

import numpy as np
import scipy as sp

class extract_data():
    def __init__(self, frame, geo):
        self.geo = geo
        self.frame = frame
        mctree = frame["I3MCTree"]
        self.primary = mctree.primaries

        self.daughter = dataclasses.I3MCTree.first_child(mctree, self.primary[0].id)

        self.primary_energy = self.primary[0].energy

        if self.daughter.type == 11 or self.daughter.type == -11:
            self.secondary_energy = self.daughter.energy

        else:
            self.secondary_energy = self.primary_energy - self.daughter.energy #daughter is the neutrino that lost energy in NC case

        self.inelasticity = (self.primary_energy - self.daughter.energy)/self.primary_energy

        #true position
        self.px, self.py, self.pz = self.primary[0].pos

        #true time
        self.primary_time = self.primary[0].time

        self.counter = frame["Counter"].value

        #true zenith and azimuth
        self.theta = self.daughter.dir.zenith
        self.phi = self.daughter.dir.azimuth

        #reconstructed position
        self.rx, self.ry, self.rz = frame['graphnet_dynedge_position_reconstruction_position_x_pred'].value, frame['graphnet_dynedge_position_reconstruction_position_y_pred'].value, frame['graphnet_dynedge_position_reconstruction_position_z_pred'].value

        #reconstructed time
        self.reco_vertex_time = frame["reco_vertex_time"].value
        
        self.reco_energy = frame["graphnet_dynedge_energy_reconstruction_energy_pred"].value

        #reconstructed zenith
        self.reco_dir_x = frame['graphnet_dynedge_direction_reconstruction_dir_x_pred'].value
        self.reco_dir_y = frame['graphnet_dynedge_direction_reconstruction_dir_y_pred'].value
        self.reco_dir_z = frame['graphnet_dynedge_direction_reconstruction_dir_z_pred'].value

        #All the pulses in the event
        pulses = frame["SplitInIcePulses_dynedge_v2_Pulses"]
        self.hits = pulses.apply(frame)
        self.qTot = sum([hit.charge for entry in self.hits for hit in entry.data()])

    def get_om_info(self, variable_type):
        if variable_type == 'true':
            #unit vector for particle direction
            ex = -np.sin(self.theta)*np.cos(self.phi)
            ey = -np.sin(self.theta)*np.sin(self.phi)
            ez = -np.cos(self.theta)
            suffix = '_true'

        elif variable_type == 'reco':
            #unit vector for particle direction
            ex, ey, ez = -self.reco_dir_x, -self.reco_dir_y, -self.reco_dir_z

            self.daughter.pos = dataclasses.I3Position(self.rx, self.ry, self.rz)
            self.daughter.time = self.reco_vertex_time #TODO change this to reco time
            suffix = '_reco'
 
        else:
            raise ValueError(f"Unknown variable_type: {variable_type}")

        map_names = ['om_counter_map',
                     'primary_energy_map',
                     'secondary_energy_map',
                     'om_x_pos_map',
                     'om_y_pos_map',
                     'om_z_pos_map',
                     'hit_time_map',
                     'hit_charge_map',
                     'distance_map',
                     'tRes_map',
                     'angle_map',
                     'om_vertex_vector_x_map',
                     'om_vertex_vector_y_map',
                     'om_vertex_vector_z_map'
                    ]


        #for name in map_names:
            #setattr(self, name + suffix, dataclasses.I3MapKeyVectorDouble())

         
        om_prop = []
       
        for entry in self.hits:
            if self.geo.omgeo.get(entry.key()).omtype.name == 'mDOM':
                #print('hit length', len(entry.data())) 
                #position on the om
                omgeo_pos = self.geo.omgeo.get(entry.key()).position
                ox, oy, oz = omgeo_pos

                #Get Angular Distribution
                #vector b/w particle and OM
                hx = self.daughter.pos.x - ox
                hy = self.daughter.pos.y - oy
                hz = self.daughter.pos.z - oz

                #dot product
                s = ex*hx + ey*hy + ez*hz

                angle = np.arccos(-s/(np.sqrt(hx**2+hy**2+hz**2)))

                #distance between vertex and om
                d = np.sqrt((self.daughter.pos.x-ox)**2+(self.daughter.pos.y-oy)**2+(self.daughter.pos.z-oz)**2)

                for hit in entry.data():
 
                    #Get Time Residuals
                    time_residual = I3Calculator.time_residual(self.daughter, omgeo_pos,
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
                    
                    om_prop.append((self.counter, self.primary_energy, self.secondary_energy, ox, oy, oz, hit.time, hit.charge, d, time_residual, angle, hx, hy, hz))
               
        values_array = list(map(np.array, zip(*om_prop)))
        #value_indices = range(0, len(om_prop))

        #for name, idx in zip(map_names, value_indices):
        for idx, name in enumerate(map_names):
            #map_obj = getattr(self, name + suffix)
            #map_obj.update({
            #entry.key(): dataclasses.I3VectorDouble(values_array[idx])
            #})

            self.frame[name + suffix] = dataclasses.I3VectorDouble(values_array[idx]) #map_obj

        #print(self.frame.keys())            
        
        #print('length', len(om_prop))
            
        return map(np.array, zip(*om_prop))

    def beta_n(self, power, coeff, array):
        legendre = sp.special.eval_legendre(power, array)
        beta = coeff*sum(legendre)
        return beta 

    def get_beta_values(self, ex, ey, ez):
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

            return self.beta_n(1, coeff, angle_arr), self.beta_n(2, coeff, angle_arr), self.beta_n(3, coeff, angle_arr), self.beta_n(4, coeff, angle_arr), self.beta_n(5, coeff, angle_arr)

        else:
            return [float('nan')] * 5

    def get_cog(self, charge, x_arr, y_arr, z_arr):
        if len(charge) != 0:
            #weighted averae to get the CoG vertex
            self.total_charge = sum(charge)
            weighted_x = sum(x_arr*charge)/self.total_charge
            weighted_y = sum(y_arr*charge)/self.total_charge
            weighted_z = sum(z_arr*charge)/self.total_charge

        return weighted_x, weighted_y, weighted_z

    def get_charge(self):
        '''
        returns the total mDOM charge and the total charge in all OMs
        '''
        return self.total_charge, self.qTot
         
def process_frame(frame, geo):

    extract = extract_data(frame, geo)
    
    #true om level info

    om_counter, om_penergy, om_senergy, ox, oy, oz, hit_time, hit_charge, distance, time_residual, angle, hx, hy, hz = extract.get_om_info('true')
    beta1, beta2, beta3, beta4, beta5 = extract.get_beta_values(hx, hy, hz)
    cog_x, cog_y, cog_z = extract.get_cog(hit_charge, ox, oy, oz) #corrected this oz was initially ox
    mDOM_total_charge, all_om_total_charge = extract.get_charge()

    #reco om level info
    om_counter_reco, om_penergy_reco, om_senergy_reco, ox_reco, oy_reco, oz_reco, hit_time_reco, hit_charge_reco, distance_reco, time_residual_reco, angle_reco, hx_reco, hy_reco, hz_reco = extract.get_om_info('reco')
    beta1_reco, beta2_reco, beta3_reco, beta4_reco, beta5_reco = extract.get_beta_values(hx_reco, hy_reco, hz_reco)
    
    #Save event level features
    event_features = {
        "EventCounter": extract.counter,
        "PrimaryEnergy": extract.primary_energy,
        "SecondaryEnergy": extract.secondary_energy,
        "RecoEnergy": extract.reco_energy,
        "TrueTime": extract.primary_time,
        "RecoVertexTime": extract.reco_vertex_time,
        "Zenith": extract.theta,
        "Azimuth": extract.phi,
        "RecoDirX": extract.reco_dir_x,
        "RecoDirY": extract.reco_dir_y,
        "RecoDirZ": extract.reco_dir_z,
        "TrueX": extract.px,
        "TrueY": extract.py,
        "TrueZ": extract.pz,
        "RecoX": extract.rx,
        "RecoY": extract.ry,
        "RecoZ": extract.rz,
        "Inelasticity": extract.inelasticity,
        "CogX": cog_x,
        "CogY": cog_y,
        "CogZ": cog_z,
        "TotalmDOMCharge": mDOM_total_charge,
        "TotalCharge": all_om_total_charge,
        "Beta1": beta1,
        "Beta2": beta2,
        "Beta3": beta3,
        "Beta4": beta4,
        "Beta5": beta5,
        "Beta1Reco": beta1_reco,
        "Beta2Reco": beta2_reco,
        "Beta3Reco": beta3_reco,
        "Beta4Reco": beta4_reco,
        "Beta5Reco": beta5_reco,
    }

    for key, value in event_features.items():
        frame[key] = dataclasses.I3Double(value)

    #print(frame.keys())

    return True

def check_mdom_hits(frame):
    '''
    Check if there are any mDOM hits in the frame. If no mDOM hits throw the frame
    '''
    hits = frame["SplitInIcePulses_dynedge_v2_Pulses_mDOMs_Only"]

    if len(hits) != 0:
        return True
    else:
        return False

def create_dataset(infiles, outfile):

    gcd_file = dataio.I3File('/home/akatil/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2')
    # Load the geometry information from a GCD file
    f_geo = dataio.I3File(gcd_file)
    geo_frame = f_geo.pop_frame(icetray.I3Frame.Geometry)
    geo = geo_frame['I3Geometry']
    
    keys = ["EventCounter", "PrimaryEnergy", "SecondaryEnergy","RecoEnergy", 
            "TrueTime", "RecoVertexTime", 
            "Zenith", "Azimuth", "RecoDirX", "RecoDirY", "RecoDirZ", 
            "TrueX", "TrueY", "TrueZ", "RecoX", "RecoY", "RecoZ",
            "Inelasticity", "CogX", "CogY", "CogZ", "TotalmDOMCharge", "TotalCharge",
            "Beta1", "Beta2", "Beta3", "Beta4", "Beta5", "Beta1Reco", "Beta2Reco", "Beta3Reco", "Beta4Reco", "Beta5Reco",
            'om_counter_map_true','primary_energy_map_true', 'secondary_energy_map_true',
            'om_x_pos_map_true', 'om_y_pos_map_true', 'om_z_pos_map_true', 'hit_time_map_true', 'hit_charge_map_true', 'distance_map_true', 'tRes_map_true',
            'angle_map_true', 'om_vertex_vector_x_map_true', 'om_vertex_vector_y_map_true',' om_vertex_vector_z_map_true',
            'om_counter_map_reco','primary_energy_map_reco', 'secondary_energy_map_reco',
            'om_x_pos_map_reco', 'om_y_pos_map_reco', 'om_z_pos_map_reco', 'hit_time_map_reco', 'hit_charge_map_reco', 'distance_map_reco', 'tRes_map_reco',
            'angle_map_reco', 'om_vertex_vector_x_map_reco', 'om_vertex_vector_y_map_reco',' om_vertex_vector_z_map_reco']

    # Create an instance of the I3Tray for data processing
    tray = I3Tray()

    # Add an I3Reader module to read input files
    tray.AddModule('I3Reader', 'reader', FilenameList=infiles)

    #extract = extract_data(frame, geo)
    #Remove frames with no mDOM hits
    tray.AddModule(check_mdom_hits, 'mdom_hits',
                   streams=[icetray.I3Frame.Physics])

    # Add the 'process_frame' module to process the data
    tray.AddModule(process_frame, 'process',
                   geo = geo,
                   Streams=[icetray.I3Frame.Physics])

    # Add an I3TableWriter module to write the dataset to an output file
    tray.AddModule(I3TableWriter, 'I3TableWriter',
                   keys = keys,
                   TableService=I3HDFTableService(outfile),
                   SubEventStreams=['InIceSplit'],                   
                   BookEverything=False)

    # Execute the processing pipeline
    tray.Execute()

    # Finish processing and save the dataset
    tray.Finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get necessary variables for analysis.")
    parser.add_argument("--dataset", type=str, default='CC', help="CC or NC")
    parser.add_argument("--batch", type=str, default='1', help="batch number of the folder")

    args = parser.parse_args()

    dataset = args.dataset
    batch = args.batch

    filelist = sorted(glob(f'/data/user/akatil/electron_neutrino/for_real/dataset_complete/batches/{dataset}/batch_{batch}/*.i3.zst'))

    outloc = f'/data/user/akatil/electron_neutrino/for_real/dataset_complete/variables/{dataset}/batch_{batch}/'
    os.makedirs(outloc, exist_ok=True)

    for f in filelist[0:10]:
        fname=f.split('/')[-1]
        #print (f"Processing file: {fname}")
        outfile=outloc+fname.strip(".i3.zst")+".h5"
        #outfile=outloc+fname.removesuffix(".i3.zst")+".h5"
        #outfile=outloc+fname
        #print([f])
        #print (f"Processing file: {outfile}")
        create_dataset([f], outfile)
