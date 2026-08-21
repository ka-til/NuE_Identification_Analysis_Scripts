from icecube.icetray import I3Tray
from icecube import icetray, dataio
from nue_classification_module import NuEClassificationModule

tray = I3Tray()

tray.AddModule("I3Reader", "reader", FilenameList=["/data/sim/IceCubeUpgrade/genie/level4_queso_v01_with_pos_reco/120029/upgrade_genie_level4_queso_120029_000000.i3.zst"])

tray.AddModule(
    NuEClassificationModule, "nue_classification", GCDFile="/home/akatil/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2"
)

tray.AddModule("I3Writer", "writer", 
               DropOrphanStreams=[icetray.I3Frame.DAQ],
               FileName="nue_example.i3.zst")

tray.Execute()
tray.Finish()
