from icecube import dataclasses, dataio, icetray, simclasses
import numpy as np
from glob import glob
from icecube.phys_services import I3Calculator


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

def get_mc_tree_prop(filelist, ux, uy, uz):
    '''
    This function is used to study the mc_tree properties

    Get the following properties:
    -primary energy
    -secondary energy
    -azimuth
    -zenith
    -distance between mean position and primary position
    -total charge
    -total charge in mDOMs
    '''
    
    data = []
    
    for file in filelist:
        infile=dataio.I3File(file)
        while(infile.more()):
            frame = infile.pop_frame()
            if frame.Stop == icetray.I3Frame.Physics:
                #get primary and daughter particle
                mctree = frame["I3MCTree"]
                primary = mctree.primaries
                daughter = dataclasses.I3MCTree.first_child(mctree, primary[0].id)
                
                #get secondary energy, for NC the energy lost is by the neutrino is taken(energy of the total hadronic showers)
                if daughter.type == 11 or daughter.type == -11:
                    energy_transfer = daughter.energy
                else:
                    energy_transfer = primary[0].energy - daughter.energy
                
                #get distance between vertex of interaction and mean position of upgrade
                px, py, pz = primary[0].pos
                distance = np.sqrt((ux - px)**2+(uy - py)**2+(uz - pz)**2)
                
                inelasticity = (primary[0].energy - daughter.energy)/primary[0].energy

                #get total charge from all the doms and only mdomd
                pulses = frame["SplitInIcePulses_dynedge_v2_Pulses"]
                hits = pulses.apply(frame)
                qTot = sum([hit.charge for entry in hits for hit in entry.data()])

                hits_m = frame["SplitInIcePulses_dynedge_v2_Pulses_mDOMs_Only"] #charge cut on mDOM only
                qTot_m = sum([hit.charge for entry in hits_m for hit in entry.data()])

                data.append((
                primary[0].energy, energy_transfer, primary[0].dir.azimuth,
                np.cos(primary[0].dir.zenith), distance, qTot, qTot_m, inelasticity
                ))

    #returns seven numpy array with data or seven empty NumPy arrays to avoid an error.
    return map(np.array, zip(*data)) if data else (np.array([]),) * 8

def get_mc_tree_prop_for_beta(filelist, ux, uy, uz):
    '''
    This function is used to study the mc_tree properties

    Get the following properties:
    -primary energy
    -secondary energy
    -azimuth
    -zenith
    -distance between mean position and primary position
    -beta1
    -angle
    '''
    
    data = []

    length = 0
    
    for file in filelist:
        infile=dataio.I3File(file)
        if length%100 == 0:
            print(length)
        length += 1
        while(infile.more()):
            frame = infile.pop_frame()
            if frame.Stop == icetray.I3Frame.Physics:
                #get primary and daughter particle
                mctree = frame["I3MCTree"]
                primary = mctree.primaries
                daughter = dataclasses.I3MCTree.first_child(mctree, primary[0].id)
                
                #get secondary energy, for NC the energy lost is by the neutrino is taken(energy of the total hadronic showers)
                if daughter.type == 11 or daughter.type == -11:
                    energy_transfer = daughter.energy
                else:
                    energy_transfer = primary[0].energy - daughter.energy
                
                #get distance between vertex of interaction and mean position of upgrade
                px, py, pz = primary[0].pos
                distance = np.sqrt((ux - px)**2+(uy - py)**2+(uz - pz)**2)
                
                inelasticity = (primary[0].energy - daughter.energy)/primary[0].energy

                #get total charge from all the doms and only mdomd
                #qtot = frame["SplitInIcePulses_dynedge_v2_PulsesHitStatistics"].value

                beta1 = frame["Beta1Reco"].value
                angle = frame["MedianAngle"].value
                pid = frame['graphnet_dynedge_track_classification_track_pred'].value
                
                data.append((daughter.type,
                primary[0].energy, energy_transfer, primary[0].dir.azimuth,
                np.cos(primary[0].dir.zenith), distance, inelasticity, beta1, angle, pid
                ))

    #returns seven numpy array with data or seven empty NumPy arrays to avoid an error.
    return map(np.array, zip(*data)) if data else (np.array([]),) * 10
