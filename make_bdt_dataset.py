import h5py
import numpy as np
import matplotlib.pyplot as plt
from icecube import icetray, dataio, phys_services                                                                        
from icecube import dataclasses  
from matplotlib.colors import LogNorm
from scipy.stats import ks_2samp
import tables
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import pearsonr
from scipy.stats import iqr
from collections import defaultdict
import pandas as pd
import os

def get_var_summary(f, var_type):

    if var_type == 'time_residual':
        var = f.root.OMMapReco.col('time_residual') 
    elif var_type == 'distance':
        var = f.root.OMMapReco.col('distance')
    elif var_type == 'angle':
        var = f.root.OMMapReco.col('angle')
    else:
        raise ValueError("Variable type you passed is not valid.")

    # Reading columns
    event_ids = f.root.EventCounter.col('value')
    om_counter = f.root.OMMapReco.col('om_counter')
    energies = f.root.OMMapReco.col('om_renergy')
    charges = f.root.OMMapReco.col('om_total_charge')

    if var_type == 'angle':
        var = np.cos(np.asarray(var))
    else:
        var = np.asarray(var)

    event_ids = np.asarray(event_ids)

    if var_type == 'time residual':
        bool_ = (var>=min_val)&(var<=max_val)
        
        var = var[bool_]
        om_counter = np.asarray(om_counter)[bool_]
        energies = np.asarray(energies)[bool_]
        charges = np.asarray(charges)[bool_]
    else:
        om_counter = np.asarray(om_counter)
        energies = np.asarray(energies)
        charges = np.asarray(charges)
        
    var_summ = []
    
    index_map = defaultdict(list) #making a default dictionary is truly a game changer it is very fast
    for idx, val in enumerate(om_counter):
        index_map[val].append(idx)
        #print(index_map)
    
    # Iterate once
    for counter, event_id in enumerate(event_ids):
        indices = index_map[event_id]
        var_vals = var[indices]
        if len(energies[indices]) != 0:
            #energy_val = energies[indices][0]  # first energy for that event
            #charge_val = charges[indices][0]
    
            var_summ.append((np.mean(var_vals), np.std(var_vals), np.median(var_vals), iqr(var_vals), np.sum(var_vals), 
                               kurtosis(var_vals), skew(var_vals)))#, energy_val, charge_val))
        
        if counter % 10000 == 0:
            print(counter)

    return map(np.array, zip(*var_summ))

def calculate_zenith_azimuth(x, y, z):

    r = np.sqrt((x)**2 + (y)**2 + (z)**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)

    return theta, phi
    

def make_dataset(f, interaction_type = 1):
    tres_mean, tres_std, tres_median, tres_iqr, tres_sum, tres_kurtosis, tres_skew = get_var_summary(f, 'time_residual')
    dist_mean, dist_std, dist_median, dist_iqr, dist_sum, dist_kurtosis, dist_skew = get_var_summary(f, 'distance')
    ang_mean, ang_std, ang_median, ang_iqr, ang_sum, ang_kurtosis, ang_skew = get_var_summary(f, 'angle')

    cogDist = np.sqrt((f.root.RecoX.col('value')-f.root.CogX.col('value'))**2+
                    (f.root.RecoY.col('value')-f.root.CogY.col('value'))**2+ 
                    (f.root.RecoZ.col('value')-f.root.CogZ.col('value'))**2)

    reco_zenith, reco_azimuth = calculate_zenith_azimuth(f.root.RecoDirX.col('value'), f.root.RecoDirY.col('value'), f.root.RecoDirZ.col('value'))

    if interaction_type == 0:
        index = np.zeros(len(f.root.EventCounter.col('value')))
        energy = f.root.PrimaryEnergy.col('value') - f.root.SecondaryEnergy.col('value') 
    elif interaction_type == 1:
        index = np.ones(len(f.root.EventCounter.col('value')))
        energy = f.root.PrimaryEnergy.col('value')
    else:
        index = np.ones(len(f.root.EventCounter.col('value')))
        index = index*2
        energy = f.root.PrimaryEnergy.col('value')
    #else:
        #index = np.ones(len(f.root.EventCounter.col('value')))
        #energy = f.root.PrimaryEnergy.col('value')

    data = {"Label": index,
        "EventCounter": f.root.EventCounter.col('value'),
        "RecoEnergy": f.root.RecoEnergy.col('value'),
        "CogDistance": cogDist,
        "TotalmDOMCharge": f.root.TotalmDOMCharge.col('value'),
        "TotalCharge": f.root.TotalCharge.col('value'),
        "Beta1Reco": f.root.Beta1Reco.col('value'),
        "Beta2Reco": f.root.Beta2Reco.col('value'),
        "Beta3Reco": f.root.Beta3Reco.col('value'),
        "Beta4Reco": f.root.Beta4Reco.col('value'),
        "Beta5Reco": f.root.Beta5Reco.col('value'),
        "tres_mean": tres_mean, 
        "tres_std": tres_std, 
        "tres_median": tres_median, 
        "tres_iqr": tres_iqr, 
        "tres_sum": tres_sum, 
        "tres_kurtosis": tres_kurtosis, 
        "tres_skew": tres_skew, 
        #"tres_energy": tres_energy, 
        #"tres_charge_val": tres_charge_val,
        "dist_mean": dist_mean, 
        "dist_std": dist_std, 
        "dist_median": dist_median, 
        "dist_iqr": dist_iqr, 
        "dist_sum": dist_sum, 
        "dist_kurtosis": dist_kurtosis, 
        "dist_skew": dist_skew, 
        #"dist_energy": dist_energy, 
        #"dist_charge_val": dist_charge_val,
        "ang_mean": ang_mean, 
        "ang_std": ang_std, 
        "ang_median": ang_median, 
        "ang_iqr": ang_iqr, 
        "ang_sum": ang_sum, 
        "ang_kurtosis": ang_kurtosis, 
        "ang_skew": ang_skew, 
        #"ang_energy": ang_energy, 
        #"ang_charge_val": ang_charge_val,
        "TrueZenith": np.cos(f.root.Zenith.col("value")),
        "TrueEnergy": energy,
        "RecoZenith": np.cos(reco_zenith),
        }

    return data

def make_pandas_dataframe():

    dfs = []

    for i in range(0, 4):

        print(f'merged cc and nc files, i is {i}')
        fCC = tables.open_file(f'/data/user/akatil/electron_neutrino/for_real/dataset_complete/merger/merged_CC_{i}.h5', mode='r')
        fNC = tables.open_file(f'/data/user/akatil/electron_neutrino/for_real/dataset_complete/merger/merged_NC_{i}.h5', mode='r')
        fMUON= tables.open_file(f'/data/user/akatil/electron_neutrino/for_real/dataset_complete/merger/merged_muon_neutrino_{i}.h5', mode='r')
        
        dataCC = make_dataset(fCC, 1)
        dfCC = pd.DataFrame(dataCC)
        dfs.append(dfCC)
        
        dataNC = make_dataset(fNC, 0)
        dfNC = pd.DataFrame(dataNC)
        dfs.append(dfNC)

        dataMUON = make_dataset(fMUON, 2)
        dfMUON = pd.DataFrame(dataMUON)
        dfs.append(dfMUON)

    combined_df = pd.concat(dfs)
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return shuffled_df


all_dfs = make_pandas_dataframe()
#all_dfs.to_hdf('../dataset_complete/BDT/BDT_all.h5', key='data', mode='w')
#all_dfs.to_hdf('../dataset_complete/BDT/BDT_all_updated.h5', key='data', mode='w')
all_dfs.to_hdf('/data/user/akatil/electron_neutrino/for_real/dataset_complete/BDT/BDT_all_updated_with_muon.h5', key='data', mode='w')


