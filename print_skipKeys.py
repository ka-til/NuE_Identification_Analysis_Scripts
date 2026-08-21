from icecube import dataclasses, dataio, icetray, simclasses
import numpy as np
from glob import glob

filelist = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/120028/upgrade_genie_level4_queso_120028_000*.i3.zst'))

daq_required_keys = ['TimeShift', 'I3MCTree', 'I3EventHeader', 'I3GenieSystWeightDict', 'I3GenieInfo', 'I3GenieResult', 'I3MCWeightDict']

physics_required_keys = ['TimeShift', 'I3MCTree', 'I3EventHeader', 'I3GenieSystWeightDict', 'I3GenieInfo', 'I3GenieResult', 'I3MCWeightDict',
                'SplitInIcePulses_dynedge_v2_Pulses', 'SplitInIcePulses_dynedge_v2_Pulses_mDOMs_Only', 
                'SplitInIcePulses_dynedge_v2_Pulses_dEggs_Only',
                'graphnet_dynedge_energy_reconstruction_energy_pred',
                'graphnet_dynedge_track_classification_track_pred',
                'graphnet_dynedge_direction_reconstruction_dir_z_pred',
                'graphnet_dynedge_direction_reconstruction_dir_y_pred',
                'graphnet_dynedge_direction_reconstruction_dir_x_pred',
                'graphnet_dynedge_direction_reconstruction_direction_kappa',
                'graphnet_dynedge_zenith_reconstruction_zenith_pred',
                'SplitInIcePulsesUpgradeHitMultiplicity']

for file in filelist[0:1]:
    infile=dataio.I3File(file)
    while(infile.more()):
        frame = infile.pop_frame()
        if frame.Stop == icetray.I3Frame.Physics:
            physics_keys = frame.keys()
            physics_skip_keys = list(set(physics_required_keys) ^ set(physics_keys))
        if frame.Stop == icetray.I3Frame.DAQ:
            daq_keys  = frame.keys()
            daq_skip_keys = list(set(daq_required_keys) ^ set(daq_keys))

print(physics_skip_keys)
