#Import necessary modules
from icecube import icetray, dataio, phys_services
from icecube import dataclasses
from icecube.icetray import I3Tray
from glob import glob
import sys
import argparse
import numpy as np

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

def impose_cuts(frame, interaction_type, mean_ux, mean_uy):
    '''
    This function imposes vertex and inelasticity cuts. 
    impose_cuts imposes a relaxed vertex cut.

    Arguments
    frame: DAQ or Physics frames
    interaction type: 1 corresponds to Charged Current Electron Neutrino(electromagnetic showers). 
    0 corresponds to hadronic showers(Using Neutral current events for now).
    2 Corresponds to muon neutrinos (Only considering the events that fail to be classified using GraphNeT)
    mean_ux, mean_uy: mean x and y position of Upgrade optical module positions.

    Returns a boolean True if the frame is passed. False if the frame does not fulfill the required criteria
    
    '''
    mctree = frame['I3MCTree']
    primary = mctree.primaries
        
    px, py, pz = primary[0].pos
    
    daughter = dataclasses.I3MCTree.first_child(mctree, primary[0].id)

    #Muon Neutrinos
    if interaction_type == 2:
        if daughter.type == 13 or daughter.type == -13:
            if primary[0].energy <= 100: 
                if px <= mean_ux + 60 and px >= mean_ux - 60:
                    if py <= mean_uy + 60 and py >= mean_uy - 60:
                        if frame['graphnet_dynedge_track_classification_track_pred'].value <= 0.6: 
                            frame["Relaxed_Vertex"] = dataclasses.I3Double(1) #Introducing this to differentiate between relaxed and strict vertex in the analysis.
                            return True
    
    #Electromagnetic Showers
    if interaction_type == 1: 
        if daughter.type == 11 or daughter.type == -11:
            if primary[0].energy <= 100: #Only selecting events less than 100 GeV because the CrossOverEnergy for hadrons is 100 GeV.
                #this means that over 100 GeV the showers are parametrized.
                #EM showers are all parametrized but these parametrizations are very well modeled. 
                #The simulations are made using the fact that cascades have extensions instead of a point source.
                if px <= mean_ux + 60 and px >= mean_ux - 60:
                    if py <= mean_uy + 60 and py >= mean_uy - 60:
                        y = (primary[0].energy - daughter.energy)/primary[0].energy
                        if y <= 0.1: #setting it to 0.1. Only want the electromagnetic showers
                            frame["Relaxed_Vertex"] = dataclasses.I3Double(1) #Introducing this to differentiate between relaxed and strict vertex in the analysis.
                            return True
                                
    #Hadronic Showers
    if interaction_type == 0: 
        if daughter.type == 12 or daughter.type == -12 or daughter.type == 14 or daughter.type == -14 or daughter.type == 16 or daughter.type == -16: 
            if primary[0].energy <= 100:
                if px <= mean_ux + 60 and px >= mean_ux - 60:
                    if py <= mean_uy + 60 and py >= mean_uy - 60:
                        frame["Relaxed_Vertex"] = dataclasses.I3Double(1)
                        return True
    return False

def strict_vertex(frame, mean_x, mean_y, mean_z):
    '''
    This function imposes a strict vertex cut. The events are almost inside the Upgrade.

    Arguments
    frame: DAQ or Physics frames
    mean_x, mean_y, mean_z: mean x, y, and z positions of Upgrade optical module.

    Returns a boolean True if the frame is passed. False if the frame does not fulfill the required criteria
    
    '''

    mctree = frame["I3MCTree"]
    primary = mctree.primaries
    primary_position = primary[0].pos

    px, py, pz = primary[0].pos

    if pz <= mean_z+150 and pz >= mean_z-150: #previously no cut in z
        if px <= mean_x+50 and px >= mean_x-50:
            if py <= mean_y+40 and py >= mean_y-40:
                frame["Strict_Vertex"] = dataclasses.I3Double(1) 
                return True #only strict vertex

    ################# we want all the frames of relaxed vertex as well, getting rid of relaxed cut to reduce file size ###############           
    return False #True #return False

def create_dataset(infile, outfile, interaction_type, mean_ux, mean_uy, mean_uz):
    '''
    This function creates the final dataset. It skips some keys not used for further analysis

    Arguments
    infile: Input file 
    outfile: Output file
    interaction_type: 0 (Hadronic Showers), 1 (Electromagnetic Showers) or 2 (Muon Neutrinos).
    mean_x, mean_y, mean_z: mean x, y, and z positions of Upgrade optical module.

    Returns a boolean True if the frame is passed. False if the frame does not fulfill the required criteria
    
    '''
    tray = I3Tray()
    # Read the file
    tray.AddModule('I3Reader', 'reader', FilenameList=infile)

    # Impose the Energy, Inelasticity and Relaxed Vertex Cut
    tray.AddModule(impose_cuts, 'cuts',
                   interaction_type = interaction_type,
                   mean_ux = mean_ux,
                   mean_uy = mean_uy,
                   Streams=[icetray.I3Frame.Physics])
                   #Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics])

    # Impose Strict Vertex Cut
    tray.AddModule(strict_vertex, 'vertex cut',
                   mean_x = mean_ux,
                   mean_y = mean_uy,
                   mean_z = mean_uz,
                   Streams=[icetray.I3Frame.Physics])
                   #Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics])
    
    skip_keys = ['SplitInIcePulses_TruthFlags',
                 'TWRTVetoSeries_ICVetoCleanedKeys',
                 'DeepCoreUpgradeFilter_13',
                 'I3MCPESeriesMapNoise_mDOM',
                 'QuesoL3_Vars_cleaned_vertexZ',
                 'graphnet_dynedge_neutrino_classification_neutrino_pred',
                 'start_time',
                 'I3MCPESeriesMapNoise_DEgg',
                 'I3MCPESeriesMap_mDOM',
                 'L4_VetoCharge',
                 'TWRTVetoSeries',
                 'L4_ToIParams',
                 'graphnet_dynedge_neutrino_vs_muon_L4_neutrino_pred',
                 'NEvPerFile',
                 'QuesoL3_Vars_uncleaned_length',
                 'I3PhotonSeriesMap',
                 'SplitInIcePulses_dynedge_v2_Pulses_pDOMs_Only',
                 'QuesoL4_Vars_graphnet_L4_neutrino_pred',
                 'DeepCoreFilter_13',
                 'QuesoL3_Vars_cleaned_num_hit_modules',
                 'QuesoL4_Bool',
                 'SplitInIcePulses_GraphSage_AuxData_round_charge',
                 'BadOM1',
                 'SplitInIcePulses_dynedge_v2_Predictions',
                 'SplitInIcePulses_GraphSage_AuxData_uncleaned_pulse_map',
                 'SplitIceCubePulses',
                 'ContainmentFlags',
                 'I3MCPESeriesMapWithNoise_PDOM',
                 'SplitInIcePulses_GraphSage_Predictions',
                 'SplitInIcePulses_dynedge_v2_PulsesUpgradeHitMultiplicity',
                 'L4_Updown_ratio',
                 'L4_FiducialCharge',
                 'IceCubePulsesTimeRange',
                 'L4_muon_firstBDT_score',
                 'I3MCPulseSeriesMap_PDOM',
                 'SplitInIcePulsesTimeRange',
                 'SaturationTimes',
                 'L4_3Zones_median_slope',
                 'SplitInIcePulses_GraphSage_AuxData_model_path',
                 'L4_VetoFiducial_RatioCharge',
                 'SplitI3RecoPulseSeriesMapGen2',
                 'I3Triggers',
                 'I3RecoPulseSeriesMap_mDOM',
                 'graphnet_dynedge_zenith_reconstruction_zenith_kappa',
                 'L4_muon_secondBDT_score',
                 'MCInIcePrimary',
                 'QuesoL3_Bool',
                 'SplitInIcePulses_GraphSage_Pulses_threshold',
                 'L4_ToI',
                 'SplitInIcePulses_GraphSage_Pulses',
                 'I3MCPESeriesMapNoise_IceCube',
                 'SplitInIcePulsesTWSRT',
                 'QuesoL3_Vars_cleaned_length',
                 'L4_first_hlc',
                 'QuesoL4_Vars_L4_muon_firstBDT_score',
                 'SplitIceCubePulsesTimeRange',
                 'I3MCPESeriesMapWithNoise_IceCube',
                 'I3TriggerHierarchy',
                 'L4_ToIEval2',
                 'TriggerSplitterLaunchWindow',
                 'L4_ToIEval3',
                 'SplitInIcePulses_dynedge_v2_PulsesICVetoCleanedKeys',
                 'MCExtraTruthInfo',
                 'L4_event_duration',
                 'L4_VICH_nch',
                 'DrivingTime',
                 'SplitInIcePulses_dynedge_v2_PulsesFidCleanedKeys',
                 'CalibrationErrata',
                 'I3MCPulseSeriesMap_IceCube',
                 'SplitInIcePulsesHitStatistics',
                 'I3MCPESeriesMapWithNoise_DEgg',
                 'SplitInIcePulses_dynedge_v2_PulsesHitStatistics',
                 'SplitIceCubePulsesTWSRT',
                 'TWRTVetoSeries_ICVeto',
                 'TWRTVetoSeriesTimeRange',
                 'I3MCPulseSeriesMap_mDOM',
                 'SplitIceCubePulsesTWSRT_TruthFlags',
                 'I3RecoPulseSeriesMap_DEgg',
                 'BadOM3',
                 'L4_first_hlc_rho',
                 'I3MCPESeriesMap_IceCube',
                 'L4_separation_in_cogs',
                 'SplitInIcePulses_dynedge_v2_PulsesICVeto',
                 'L4_VectDistRatio',
                 'QuesoL3_Vars_cleaned_num_hits_fid_vol',
                 'SplitInIcePulsesSRT',
                 'SplitInIcePulses_GraphSage_AuxData_n_features',
                 'SplitInIcePulsesTWSRT_TruthFlags',
                 'SplitIceCubePulses_TruthFlags',
                 'SplitInIcePulses_GraphSage_AuxData_adj_threshold',
                 'MCTimeIncEventID',
                 'CalibratedWaveformRange',
                 'L4_RTVeto250Charge',
                 'SplitInIcePulsesCleaned',
                 'QAbove200',
                 'SplitInIcePulses_GraphSage_AuxData_dataset_id',
                 'I3MCPESeriesMapNoise_PDOM',
                 'SplitI3RecoPulseSeriesMapGen2TimeRange',
                 'I3MCPESeriesMap_DEgg',
                 'SplitIceCubePulsesSRT',
                 'I3MCPulseSeriesMap_DEgg',
                 'I3RecoPulseSeriesMap_PDOM',
                 'I3MCPESeriesMap_PDOM',
                 'QuesoL4_Vars_L4_muon_secondBDT_score',
                 'RTVetoSeries250',
                 'SplitInIcePulses_dynedge_v2_PulsesFid',
                 'I3MCPESeriesMapWithNoise_mDOM',
                 'SaturatedDOMs',
                 'BadOM2',
                 'SplitInIcePulsesTWSRTTimeRange',
                 'InIceRawData']
    #Got this from a simple script to ensure only necessary frame keys are stored

    # Write selected frames to output file, skipping specified keys and dropping orphan DAQ streams
    tray.AddModule('I3Writer','writer',
                   FileName=outfile, 
                   SkipKeys=skip_keys,
                   DropOrphanStreams=[icetray.I3Frame.DAQ])

    tray.Execute()
    tray.Finish()

#This only runs when the file is directly run and not when it is imported
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process IceCube data and create a dataset.")
    parser.add_argument("--interaction", type=int, default=1, help="Interaction type, 1 is CC electron neutrino and 0 is NC Events. Interaction type 2 is Muon neutrinos")
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

    #Muon Neutrino
    if interaction_type == 2:

        filelist1 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/140'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
        #NuMu
        filelist2 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/141'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
        #NuMuBar

        filelist = filelist1 + filelist2
        outloc = outdir+'muon_neutrino/'

    #Electromagnetic Shower
    if interaction_type == 1:

        filelist1 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/120'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
        #NuE
        filelist2 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/121'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
        #NuEBar

        filelist = filelist1 + filelist2
        outloc = outdir+'CC/'

    #Hadronic Showers
    if interaction_type == 0:

        filelist1 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/120'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
        filelist2 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/121'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
        filelist3 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/140'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
        filelist4 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/141'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
        filelist5 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/160'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))
        filelist6 = sorted(glob('/data/sim/IceCubeUpgrade/genie/level4_queso_v01/161'+dataset+'/upgrade_genie_level4_queso_*.i3.zst'))

        filelist = filelist1+filelist2+filelist3+filelist4+filelist5+filelist6 #Getting NC events from all the files.
        outloc = outdir+'NC/'

    #loop through each file from the file list and create a new output file after making necessary cuts.
    for f in filelist:
        fname=f.split('/')[-1]
        #print (f"Processing file: {fname}")
        outfile=outloc+fname
        #print([f])
        #print (f"Processing file: {outfile}")
        create_dataset([f], outfile, interaction_type, ux, uy, uz)
