import sys, os
import numpy as np
from glob import glob
import argparse

# IceCube modules (some are required to produce table files)
from icecube import icetray, dataio, dataclasses, linefit

sys.path.append('/home/jpyanez/software/santa_yanez/python')
import SANTA_fit


# Adding santa
from icecube.icetray import I3Tray

def NeutrinoSeed(frame):
    mctree = frame['I3MCTree']
    frame['TrueNu'] = mctree.get_primaries()[0]
    return

def run_santa(files, outfile):

    tray = I3Tray()


    # Parameters to run the reco
    
    name = 'ana_chain'
    InputPulseSeries  = 'SplitInIcePulses_dynedge_v2_Pulses'
    LFname            = 'LineFit'
    FitResultsZenith  = 'SANTA_Track'
    FitResultsCascade = 'SANTA_Cascade'
    Interactive       = False
    D0scaling         = 7.
    santa_suffix      = '' # Use if the fit is done more than once
    Debugging         = False
    StrongFitQuality  = True
    If                = lambda f: True
    MultiStringFit    = True

    tray.AddModule('I3Reader', 'reader',
                   FilenameList = files,
                   )
    
    #tray.AddSegment(linefit.simple,"example",
    #                inputResponse = InputPulseSeries,
    #                fitName = LFname)
    
    tray.AddModule(NeutrinoSeed, 'nuseed')
    
    tray.AddModule("I3LineFit", "SANTA_Linefit_"+name,
                   InputRecoPulses =  InputPulseSeries,
                   LeadingEdge     = "FLE",
                   Name            = LFname,
                   If = If,
                   )
    
    
    tray.AddModule(SANTA_fit.SANTA_EventFits, 'SANTA_Fit_'+name,
                   Interactive        = Interactive,
                   InputPulseSeries   = InputPulseSeries,
                   FitResults_Zenith  = FitResultsZenith  + santa_suffix,
                   FitResults_Cascade = FitResultsCascade + santa_suffix,
                   MultiStringSeed    = LFname,
                   StrongFitQuality   = StrongFitQuality,
                   D0scaling          = D0scaling,
                   MultiStringFit     = MultiStringFit,
                   If =  If
                   )
    
    
    tray.AddModule('I3Writer', 'writer',
                   FileName = outfile,
                   Streams = [icetray.I3Frame.DAQ, icetray.I3Frame.Physics])
    
    #tray.AddModule('`:TrashCan','can')
    tray.Execute()
    tray.Finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process IceCube data and create a dataset.")
    parser.add_argument("--indir", type=str, default='/data/user/akatil/electron_neutrino/for_real/dataset_complete/', help="Give the input directory where files are located")
    parser.add_argument("--interaction", type=int, default=1, help="Interaction type, 1 is CC electron neutrino and 0 is NC Events")
    parser.add_argument("--dataset", type=str, default='028', help="Give the last three digits of the dataset")
    parser.add_argument("--outdir", type=str, default='/data/user/akatil/electron_neutrino/for_real/dataset_complete/', help="Give the output directory where files should be located")

    args = parser.parse_args()

    interaction_type = args.interaction
    outdir = args.outdir
    dataset = args.dataset
    indir = args.indir

    gcd_file = '/home/akatil/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2'

    if interaction_type == 1:
        filelist = sorted(glob(indir+'CC/upgrade_genie_level4_queso_*'+dataset+'_000*.i3.zst'))
        #NuE
        #filelist2 = sorted(glob(indir+'CC/upgrade_genie_level4_queso_121'+dataset+'_000*.i3.zst'))
        #NuEBar
        #filelist = filelist1 + filelist2

        outloc = os.path.join(outdir, 'CC_santa') 
        os.makedirs(outloc, exist_ok=True)  # Create the directory if it doesn't exist

    if interaction_type == 0:
        filelist = sorted(glob(indir+'NC/upgrade_genie_level4_queso_*'+dataset+'_000*.i3.zst'))
        #filelist2 = sorted(glob(indir+'NC/upgrade_genie_level4_queso_121'+dataset+'_000*.i3.zst'))
        #filelist3 = sorted(glob(indir+'NC/upgrade_genie_level4_queso_140'+dataset+'_000*.i3.zst'))
        #filelist4 = sorted(glob(indir+'NC/upgrade_genie_level4_queso_141'+dataset+'_000*.i3.zst'))
        #filelist5 = sorted(glob(indir+'NC/upgrade_genie_level4_queso_160'+dataset+'_000*.i3.zst'))
        #filelist6 = sorted(glob(indir+'NC/upgrade_genie_level4_queso_161'+dataset+'_000*.i3.zst'))

        #filelist = filelist1+filelist2+filelist3+filelist4+filelist5+filelist6
        outloc = os.path.join(outdir, 'NC_santa')
        os.makedirs(outloc, exist_ok=True)  # Create the directory if it doesn't exist
    
    for f in filelist:
        #print(f)
        files = [gcd_file, f]
        fname=f.split('/')[-1]
        outfile=outloc+'/'+fname
        run_santa(files, outfile)
