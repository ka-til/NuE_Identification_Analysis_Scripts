import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib

#Import necessary modules
from icecube import icetray, dataio, phys_services
from icecube import dataclasses
from icecube.icetray import I3Tray
from icecube.common_variables import hit_statistics
from glob import glob
import sys
import argparse
import numpy as np
import os

def bdt_score(frame):

    frame_data = {
    'RecoEnergy': frame["graphnet_dynedge_energy_reconstruction_energy_pred"].value,
    'TotalCharge': frame["SplitInIcePulses_dynedge_v2_PulsesHitStatistics"].q_tot_pulses,
    'CogDistance': frame["CogDistance"].value,
    'Beta1Reco': frame["Beta1Reco"].value,
    'tres_median': frame["MediantRes"].value,
    'tres_iqr': frame["IqrtRes"].value,
    'tres_skew': frame["SkewtRes"].value,
    'dist_iqr': frame["IqrDistance"].value,
    'dist_skew': frame["SkewDistance"].value,
    'ang_median': frame["MedianAngle"].value,
    'ang_iqr': frame["IqrAngle"].value,
    'ang_kurtosis': frame["KurtosisAngle"].value,
    'ang_skew': frame["SkewAngle"].value
    }
    
    df = pd.DataFrame(frame_data, index=[0])

    #model = joblib.load('/data/user/akatil/electron_neutrino/for_real/analysis_chain/bdt_scripts/submit_scripts/seventh_iter.joblib')

    model_nc = joblib.load('./BDT_models/nue_nc_classifier.joblib')

    model_numu = joblib.load('./BDT_models/nue_numu_classifier.joblib')
    
    if df.isnull().values.any():
        score_nc = [np.nan]
        score_numu = [np.nan]
    else:
        #score = model.predict_proba(df)[:, 1]
        score_nc = model_nc.predict_proba(df)[:, 1]
        score_numu = model_numu.predict_proba(df)[:, 1]

    #frame['NuE_BDT_classifier_score'] = dataclasses.I3Double(score[0])
    frame['NuE_NC_BDT_classifier_score'] = dataclasses.I3Double(score_nc[0])
    frame['NuE_NuMu_BDT_classifier_score'] = dataclasses.I3Double(score_numu[0])
    
def create_dataset(infile, outfile):

    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader', FilenameList=infile)
    
    tray.AddModule(bdt_score, 'bdt score',
                   Streams=[icetray.I3Frame.Physics])

    tray.AddModule('I3Writer','writer',
                   FileName=outfile,
                   DropOrphanStreams=[icetray.I3Frame.DAQ])
    
    tray.Execute()
    tray.Finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process IceCube data and create a dataset.")
    parser.add_argument("--indir", type=str, default='/data/sim/IceCubeUpgrade/genie/level4_queso_v01_with_pos_reco/120029/', help="Give the input directory where files should be located")
    #parser.add_argument("--dataset", type=str, default='120029', help="Give the last three digits of the dataset")
    parser.add_argument("--outdir", type=str, default='./', help="Give the output directory where files should be located")

    args = parser.parse_args()

    args.indir = args.indir
    outdir = args.outdir
    #dataset = args.dataset

    filelist = sorted(glob(f'{args.indir}*.i3.zst'))
    
    outloc = os.path.join(outdir, '')
    os.makedirs(outloc, exist_ok=True)

    for f in filelist:
        fname=f.split('/')[-1]
        outfile=outloc+'nue_'+fname
        create_dataset([f], outfile)



