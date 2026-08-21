# Electron Neutrino Idetification Module

This `I3Module` returns two BDT classifier scores:
* **NC Classifier (`BDT_models/nue_nc_classifier`):** A smaller score represents neutral current events, while a larger score represents charged current electron neutrino events.
* **NuMu Classifier (`BDT_models/nue_numu_classifier`):** A smaller score represents muon neutrino events, while a larger score represents charged current electron neutrino events.

**Dataset:** `/data/sim/IceCubeUpgrade/genie/level4_queso_v01_with_pos_reco/`

**IceTray environment:**
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh`

/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.9.2/env-shell.sh

**Software requirements:** `scikit-learn`. (To start with, you can use the existing `graphnet` virtual env with `scikit-learn` at `/home/akatil/graphnet_env.sh`. Use the IceTray environment mentioned above before activating the virtual environment.)

The module assumes that vertex reconstruction has been performed. If this is not done, use the script `graphnet_vertex_reco.py`. The model used here was trained with upgrade simulation that has 7 strings.

`example.py` shows the implementation of `NuEClassificationModule`.

`extract_var.py` adds necessary variables to the frame. This is implemented in the full `nue_classification_module.py`, but you can also do an independent implementation.

`apply_BDT_model_to_i3.py` is implemented in `nue_classification_module.py`, but run it as an independent module to get the BDT score. If going for an independent implementation, run `extract_var.py` first.