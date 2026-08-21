#GraphNeT needs to be installed to run this script
from glob import glob
from os.path import join
from typing import TYPE_CHECKING

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.extractors.icecube import (
     I3FeatureExtractorIceCubeUpgrade,
)
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

from graphnet.deployment.i3modules import I3InferenceModule, GraphNeTI3Deployer


features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE

#This model applies the pretrained model to get the vertex.
def main(
    input_dir: list[str],
    pretrained_model_dir: str,
    gcd_file: str,
    output_dir: str,
    dataset: str,
    num_workers: int,
   ) -> None:
    
    print("in main")
    pulsemap = 'SplitInIcePulses_dynedge_v2_Pulses'
    input_folders = join(input_dir, dataset)

    #defining path, folder and files required for I3InferenceModule and GraphNeTI3Deployer
    base_path = pretrained_model_dir
    model_config = f"{base_path}/model_config.yml"
    state_dict = f"{base_path}/state_dict.pth"
    output_folder = join(output_dir, dataset)
    gcd_file = gcd_file
    input_files = []
    #for folder in input_folders:
    #print("folder is: ", folder)
    input_files.extend(glob(join(input_folders, "*.i3.zst")))

    print(f"input directory is {input_folders} AND input file length is {len(input_files)}")
    print(f"output directory is {output_folder}")

    print("loaded necessary files, will run the deployment")
    #Configure Deployment Module
    deployment_module = I3InferenceModule(
        pulsemap = pulsemap,
        features = features,
        pulsemap_extractor = I3FeatureExtractorIceCubeUpgrade(pulsemap=pulsemap),
        model_config = model_config,
        state_dict = state_dict,
        gcd_file = gcd_file,
        prediction_columns = ["position_x_pred", "position_y_pred", "position_z_pred"],
        model_name="graphnet_dynedge_position_reconstruction",
    )

    #Construct  I3 Deployer
    deployer = GraphNeTI3Deployer(
        graphnet_modules=[deployment_module],
        n_workers=num_workers,
        gcd_file=gcd_file,
    )


    #start deployment 
    deployer.run(
        input_files = input_files,
        output_folder = output_folder,
    )

    print("finished deployment")


if __name__ == "__main__":

    #parse command-line arguments

    parser = ArgumentParser(
        description = """ Use GraphNeTI3Modules to deploy trained model with GraphNeTI3Deployer"""
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to the folders of .i3 files",
        default='/data/user/akatil/electron_neutrino/for_real/dataset_complete' #'/data/sim/IceCubeUpgrade/genie/level4_queso_v01'
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="output path to the folders of .i3 files",
        default='/data/user/akatil/electron_neutrino/for_real/dataset_complete/reconstruction' #'/data/sim/IceCubeUpgrade/genie/level4_queso_v01_with_pos_reco'
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="which dataset do you work on?",
        default='CC',
    )

    parser.add_argument(
        "--pretrained-model-dir",
        type=str,
        help="Path to the pretrained model and state dict",
        default="/data/user/akatil/electron_neutrino/for_real/submit_graphnet_vertex/cc_cascade_train_model/dev_step4_upgrade_028_with_noise_dynedge_pulsemap_v3_merger_aftercrash/dynedge_cc_cascade/",
    )

    parser.add_argument(
        "--gcd-file",
        type=str,
        help="Give the gcd file",
        default="/home/akatil/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V1.i3.bz2",
    )

    parser.add_argument(
        "--num-workers",
        type=int,   
        help="How many workers do you need, your majesty",
        default=1
    )
            
    args = parser.parse_args()

    print("now starting main")
    print(args.input_dir, args.num_workers)

    main(
        args.input_dir,
        args.pretrained_model_dir,
        args.gcd_file,
        args.output_dir,
        args.dataset,
        args.num_workers,
    )
  
   
