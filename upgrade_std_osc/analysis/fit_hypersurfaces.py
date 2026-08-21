'''
Fit simple hypersurfaces for the IceCube Upgrade

Kayla Leonard DeHolton

Adapted from OscNext fit_hypersurfaces.py from: Tom Stuttard
'''

#TODO store sys params in the HDF5 file metadata when running the i3->PISA converter

import os, copy
import numpy as np

from pisa.utils.config_parser import parse_pipeline_config
from pisa.utils.hypersurface import *

from utils.filesys_tools import make_dir
from utils.cluster.cluster import ClusterSubmitter

from analysis.upgrade_std_osc.analysis.upgrade_std_osc_analysis import configure_analysis, SAMPLE, DATA_LOADER_STAGE_KEY, ANALYSES, SETUP_SCRIPT
from analysis.common.utils.param_tools import apply_param_patches
from analysis.common.analysis.core.oscillations_analysis import remove_hypersurface_stage


#
# Globals
#

HIST_STAGE_KEY = ("utils", "hist")
KDE_STAGE_KEY = ("utils", "kde")

UPGRADE_PISA_FILE_VERSION = "level4_queso_v01"
UPGRADE_PISA_FILE_DIR = os.path.join("/data/sim/IceCubeUpgrade", "pisa", UPGRADE_PISA_FILE_VERSION)

# Define fit function args in a single place here
# Can use this every time we call this to ensure consistency
FIT_HYPERSURFACES_KW = {
    "fix_intercept" : False,
    "log" : False,
    "intercept_bounds" : None,
    "include_empty" : False,
}

# Hypersurface paths
HYPERSURFACE_SETTINGS_PATH = os.path.expandvars("../settings/discr_sys/hypersurfaces/")

#
# Neutrino steering
#


def get_neutrino_datasets(nominal_pipeline, exclude_sets=None) :
    '''
    Get all neutrino datasets, based on the steering provided

    The values for the systematics parameters in here should match the values in the following scripts:
    --> TODO 

    Args:
    '''

    #
    # Process inputs
    #

    # Default sets to exclude
    if exclude_sets is None :
        exclude_sets = []
    elif exclude_sets == False :
        exclude_sets = []

    # Init containers
    datasets = []


    #
    # Get dataset definitions
    #
    datasets.extend([

        # Nominal set
        {
            "file" : os.path.join(UPGRADE_PISA_FILE_DIR, "ICU_pisa_genie_029_%s.hdf5"%UPGRADE_PISA_FILE_VERSION),
            "dataset" : "0",
            "sys_params" : {
                "dom_eff" : 1.0,
                "icu_dom_eff" : 1.,
                "bulk_ice_scatter" : 1.,
                "bulk_ice_abs" : 1.,
            },
        },

        # IC&DC DOM efficiency
        { 
            "file" : os.path.join(UPGRADE_PISA_FILE_DIR, "ICU_pisa_genie_129_%s.hdf5"%UPGRADE_PISA_FILE_VERSION),
            "dataset" : "1",
            "sys_params" : {
                "dom_eff" : 1.1,
                "icu_dom_eff" : 1.,
                "bulk_ice_scatter" : 1.,
                "bulk_ice_abs" : 1.,
            },
        },

        { 
            "file" : os.path.join(UPGRADE_PISA_FILE_DIR, "ICU_pisa_genie_229_%s.hdf5"%UPGRADE_PISA_FILE_VERSION),
            "dataset" : "2",
            "sys_params" : {
                "dom_eff" : 0.9,
                "icu_dom_eff" : 1.,
                "bulk_ice_scatter" : 1.,
                "bulk_ice_abs" : 1.,
            },
        },

        # ICU (mdom & degg & pdom) DOM efficiency
        { 
            "file" : os.path.join(UPGRADE_PISA_FILE_DIR, "ICU_pisa_genie_329_%s.hdf5"%UPGRADE_PISA_FILE_VERSION),
            "dataset" : "3",
            "sys_params" : {
                "dom_eff" : 1.,
                "icu_dom_eff" : 1.05,
                "bulk_ice_scatter" : 1.,
                "bulk_ice_abs" : 1.,
            },
        },

        { 
            "file" : os.path.join(UPGRADE_PISA_FILE_DIR, "ICU_pisa_genie_429_%s.hdf5"%UPGRADE_PISA_FILE_VERSION),
            "dataset" : "4",
            "sys_params" : {
                "dom_eff" : 1.,
                "icu_dom_eff" : 0.95,
                "bulk_ice_scatter" : 1.,
                "bulk_ice_abs" : 1.,
            },
        },

        # Bulk Ice Scattering
        { 
            "file" : os.path.join(UPGRADE_PISA_FILE_DIR, "ICU_pisa_genie_529_%s.hdf5"%UPGRADE_PISA_FILE_VERSION),
            "dataset" : "5",
            "sys_params" : {
                "dom_eff" : 1.,
                "icu_dom_eff" : 1.,
                "bulk_ice_scatter" : 1.1,
                "bulk_ice_abs" : 1.,
            },
        },

        { 
            "file" : os.path.join(UPGRADE_PISA_FILE_DIR, "ICU_pisa_genie_629_%s.hdf5"%UPGRADE_PISA_FILE_VERSION),
            "dataset" : "6",
            "sys_params" : {
                "dom_eff" : 1.,
                "icu_dom_eff" : 1.,
                "bulk_ice_scatter" : 0.9,
                "bulk_ice_abs" : 1.,
            },
        },

        # Bulk Ice Absorption
        { 
            "file" : os.path.join(UPGRADE_PISA_FILE_DIR, "ICU_pisa_genie_729_%s.hdf5"%UPGRADE_PISA_FILE_VERSION),
            "dataset" : "7",
            "sys_params" : {
                "dom_eff" : 1.,
                "icu_dom_eff" : 1.,
                "bulk_ice_scatter" : 1.,
                "bulk_ice_abs" : 1.1,
            },
        },

        { 
            "file" : os.path.join(UPGRADE_PISA_FILE_DIR, "ICU_pisa_genie_829_%s.hdf5"%UPGRADE_PISA_FILE_VERSION),
            "dataset" : "8",
            "sys_params" : {
                "dom_eff" : 1.,
                "icu_dom_eff" : 1.,
                "bulk_ice_scatter" : 1.,
                "bulk_ice_abs" : 0.9,
            },
        },

    ])


    #
    # Remove any excluded sets
    #

    if len(exclude_sets) > 0 :

        print("Removing %i excluded datasets :" % len(exclude_sets))
        for s in exclude_sets :
            print("  %s"%s)

        datasets = [ d for d in datasets if d["dataset"] not in exclude_sets ] 


    #
    # Choose nominal dataset
    #

    # If a config file was provided, parse it
    if isinstance(nominal_pipeline, str) :
        nominal_pipeline_parsed = parse_pipeline_config(nominal_pipeline)
    else :
        nominal_pipeline_parsed = nominal_pipeline

    # Determine which dataset is being used by the nominal pipeline
    # Do this by matching the file name
    nominal_dataset, i_nominal_dataset = None, None
    nominal_events_file = nominal_pipeline_parsed[DATA_LOADER_STAGE_KEY]["events_file"]
    for i_dataset, dataset in enumerate(datasets) :
        if dataset["file"] == nominal_events_file :
            assert nominal_dataset is None, "Found multiple nominal datasets!?!"
            nominal_dataset = dataset
            i_nominal_dataset = i_dataset
    assert nominal_dataset is not None, "Nominal dataset not found"

    # Add pipeline cfg to nominal dataset def
    # Handling cases where either a file path or a pre-loaded config are passed
    if isinstance(nominal_pipeline, str) :
        nominal_dataset["pipeline_cfg_file"] = nominal_pipeline
    nominal_dataset["pipeline_cfg"] = nominal_pipeline_parsed

    # The sys datasets are then all datasets other than the nominal
    num_datasets = len(datasets)
    sys_datasets = [ dataset for i_dataset, dataset in enumerate(datasets) if i_dataset != i_nominal_dataset ]
    assert num_datasets - len(sys_datasets) == 1, "Nominal dataset not removed"
    num_datasets = len(sys_datasets)


    #
    # Done
    #

    # Report...
    print("")

    print("")

    print("Found nominal dataset :")
    for k, v in nominal_dataset.items() :
        if k != "pipeline_cfg" :
            print(f"  {k} : {v}")

    print("")

    print(f"Found {num_datasets} sys datasets : ")
    for d in sys_datasets :
        print(f"  {d['dataset']} : {d['sys_params']}")

    # Return
    return nominal_dataset, sys_datasets


# Combine differnt sub-categories
# Merging (a) nu+nubar, (b) all NC, and (c) NC and nue CC
# This must match `links` in `nuetrino_hypersurface_stage.cfg`
NEUTRINO_COMBINE_REGEX = ["nue.*_cc|.*_nc","numu.*_cc","nutau.*_cc"]
# NEUTRINO_COMBINE_REGEX = ["nue.*_cc","numu.*_cc","nutau.*_cc",".*_nc"]

def get_neutrino_fit_hypersurfaces_kw():

    # Define the systematic params, and their functional form
    neutrino_params = [
        HypersurfaceParam( name="dom_eff", func_name="linear",
                           initial_fit_coeffts=[0.],
                           coeff_prior_sigma=[5.]
                         ),
        HypersurfaceParam( name="icu_dom_eff", func_name="linear",
                           initial_fit_coeffts=[0.],
                           coeff_prior_sigma=[5.]
                         ),
        HypersurfaceParam( name="bulk_ice_abs", func_name="linear",
                           initial_fit_coeffts=[0.],
                           coeff_prior_sigma=[5.]
                         ),
        HypersurfaceParam( name="bulk_ice_scatter", func_name="linear",
                           initial_fit_coeffts=[0.],
                           coeff_prior_sigma=[5.]
                         ),
    ]

    # Steering of neutrino hypersurface fits
    neutrino_fit_hypersurfaces_kw = {
        "params" : neutrino_params,
        "combine_regex" : NEUTRINO_COMBINE_REGEX, 
    }
    neutrino_fit_hypersurfaces_kw.update(FIT_HYPERSURFACES_KW)

    return neutrino_fit_hypersurfaces_kw



#
# Helper functions
#


def get_sys_dataset_pipeline_cfg(nominal_pipeline_cfg, sys_dataset_file):
    """
    Get the pipeline config for a systematics dataset
    Do this by grabbing the nominal version and replacing the file path
    """

    # Parse the pipelien cfg if hasn't already been parsed
    if isinstance(nominal_pipeline_cfg, str):
        nominal_pipeline_cfg = parse_pipeline_config(nominal_pipeline_cfg)

    # Make a copy to use the same cfg for the systematic dataset
    sys_pipeline_cfg = copy.deepcopy(nominal_pipeline_cfg)

    # Update the file path to point to the file for the systematic dataset
    sys_pipeline_cfg[DATA_LOADER_STAGE_KEY]["events_file"] = sys_dataset_file

    return sys_pipeline_cfg


def get_pipeline_name(pipeline_cfg) :
    '''
    Get pipeline name from a cfg file
    '''

    # If a config file was provided, load it
    if isinstance(pipeline_cfg, str) :
        parsed_pipeline_cfg = parse_pipeline_config(pipeline_cfg)
    else :
        parsed_pipeline_cfg = pipeline_cfg

    # Grab the "name" specified in the pipleline config
    name = parsed_pipeline_cfg["pipeline"]["name"]

    return name


def get_hypersurface_config(pipeline_cfg, param_patches=None, keep_param_names=None, mass_ordering=None, fix_intercept=False) :
    '''
    Get the hypersurface config for the specified pipeline
    '''

    #
    # Handle inputs
    #

    # If a config file was provided, load it
    if isinstance(pipeline_cfg, str) :
        parsed_pipeline_cfg = parse_pipeline_config(pipeline_cfg)
    else :
        parsed_pipeline_cfg = pipeline_cfg


    #
    # Get hypersurface fitting config
    #

    # Grab the "name" specified in the pipleline config
    name = get_pipeline_name(parsed_pipeline_cfg)

    # Get config based on particle type
    if name in ["neutrino", "neutrinos"] :


        #
        # Neutrinos
        #

        label = "neutrino"

        # Get datasets
        nominal_dataset, sys_datasets = get_neutrino_datasets(pipeline_cfg)

        # Get fit steering
        hypersurface_fit_kw = get_neutrino_fit_hypersurfaces_kw()


    else :
        raise Exception("Cannot identify pipeline as muons or neutrinos (name is '%s') : %s" % (pipeline_cfg, name))

    # Fix intercepts (e.g. don't fit them), if requested
    if fix_intercept :
        hypersurface_fit_kw["fix_intercept"] = True


    #
    # Prepare dataset specifications for passing to fitter
    #

    # if the pipeline config is a path to a config file (e.g. hasn't already been loaded), then load it
    if "pipeline_cfg" not in nominal_dataset:
        assert "pipeline_cfg_file" in nominal_dataset
        nominal_dataset["pipeline_cfg"] = parse_pipeline_config(nominal_dataset["pipeline_cfg_file"])
    
    # Remove any existing hypersurface stage
    # e.g. don't want to run use the output of a previous hypersurface to fit the new hypersurface
    nominal_dataset["pipeline_cfg"] = remove_hypersurface_stage(nominal_dataset["pipeline_cfg"])
    
    # Remove any existing upsampling stage
    # hypersurfaces should be fit on the coarse binning
    for stage_key, stage_cfg in list(nominal_dataset["pipeline_cfg"].items()):
        if "resample" in stage_key[1]:
            del nominal_dataset["pipeline_cfg"][stage_key]
    
    # Remove any existing error fixing stage
    # The error fixing stage has a binning that is incompatible with the histograms needed for the hypersurface fits
    for stage_key, stage_cfg in list(nominal_dataset["pipeline_cfg"].items()):
        if "fix_error" in stage_key[1]:
            del nominal_dataset["pipeline_cfg"][stage_key]

    # Apply any param patches
    if param_patches is not None :
        apply_param_patches(nominal_dataset["pipeline_cfg"], param_patches)

    # Add the piplines to the systematics sets dicts, substituting the correct file
    for sys_dataset in sys_datasets:
        sys_dataset["pipeline_cfg"] = get_sys_dataset_pipeline_cfg(
            nominal_pipeline_cfg=nominal_dataset["pipeline_cfg"],
            sys_dataset_file=sys_dataset["file"],
        )


    #
    # Trim params
    #

    # Use can optionally specify a subset of hypersurface params
    # If so, only use those

    if keep_param_names is not None :

        nominal_dataset, sys_datasets, hypersurface_fit_kw["params"] = trim_params(
            keep_param_names=keep_param_names, 
            params=hypersurface_fit_kw["params"], 
            nominal_dataset=nominal_dataset, 
            sys_datasets=sys_datasets)

        
    #
    # Done
    #

    return nominal_dataset, sys_datasets, hypersurface_fit_kw, label


def get_default_tag(analysis_name, label) :
    '''
    Generate a default tag for file naming
    '''

    tag = SAMPLE + "__" + analysis_name + "__" + label

    return tag


def switch_pipeline_to_kde(pipeline_cfg) :
    '''
    Remove the histogram stage from a pipleine and instead add the kde
    stage so that a KDE can be used to generate the binned expectation.

    This gives a better estimate in the case of low statistics,
    although at the cost of losing the MC uncertainty estimate.
    '''

    #TODO This hasn't had much verification yet, and seems crazy slow

    #TODO need to enforce there is not HS stage, or handle that case

    raise Exception("This KDE function needs to remove cuts on the binning limits (as these events are needed to form the KDE)")

    pipeline_cfg = copy.deepcopy(pipeline_cfg)

    if isinstance(pipeline_cfg, str) :
        pipeline_cfg = parse_pipeline_config(pipeline_cfg)

    # Remove hist stage, and keep the settings
    assert HIST_STAGE_KEY in pipeline_cfg
    hist_stage_cfg = ( pipeline_cfg.pop(HIST_STAGE_KEY) )

    # Add KDE stage (to end of pipeline)
    # Use same binning definitions as the (now removed) hist stage
    pipeline_cfg[KDE_STAGE_KEY] = {
        "calc_mode" : hist_stage_cfg["calc_mode"],
        "apply_mode" : hist_stage_cfg["apply_mode"],
        "bw_method" : "silverman", # #TODO arg
        "oversample" : 10, #TODO arg
        "coszen_reflection" : 0.3, #TODO arg
        "stack_pid" : True, #TODO arg
    }

    return pipeline_cfg


def trim_params(keep_param_names, params, nominal_dataset, sys_datasets):
    """
    Remove some dimensions/params
    Useful for testing without having to comment out loads of code
    """

    #
    # Systematics sets
    #

    # Loop over sys sets
    new_sys_datasets = []
    for sys_dataset in sys_datasets:

        # Start from a copy
        new_sys_dataset = copy.deepcopy(sys_dataset)

        # Only keep on-axis sets for the specifed params
        on_axis = True
        for param_name, param_nom_val in list(nominal_dataset["sys_params"].items()):
            if param_name not in keep_param_names:
                if not np.isclose(
                    new_sys_dataset["sys_params"][param_name], param_nom_val
                ):
                    on_axis = False
                    break
        if on_axis:
            new_sys_datasets.append(new_sys_dataset)

        # Trim param values within this sys set
        new_sys_dataset["sys_params"] = {
            k: v
            for k, v in list(new_sys_dataset["sys_params"].items())
            if k in keep_param_names
        }

    #
    # Hypersurface parameters
    #

    # Trim params
    new_params = [
        copy.deepcopy(param) for param in params if param.name in keep_param_names
    ]

    #
    # Nominal dataset
    #

    # Start from a copy
    new_nominal_dataset = copy.deepcopy(nominal_dataset)

    # Trim param values within the nominal set
    new_nominal_dataset["sys_params"] = {
        k: v
        for k, v in list(new_nominal_dataset["sys_params"].items())
        if k in keep_param_names
    }

    #
    # Report
    #

    print(">>>>>> Trimming params >>>>>>")
    print("Keeping : %s" % keep_param_names)
    print("Nominal dataset :")
    print("  %s" % new_nominal_dataset["sys_params"])
    print("Systematics datasets :")
    for new_sys_set in new_sys_datasets :
        print("  %s" % new_sys_set["sys_params"])
    print("<<<<<< Trimming params <<<<<<")

    return new_nominal_dataset, new_sys_datasets, new_params


#
# Neutrino fit functions
#

def submit_neutrino_grid_fits(output_dir, interpolation_param_spec, pipeline_cfg=None, param_patches=None, flush_factor=1, keep_param_names=None, fix_intercept=True) :
    '''
    Prepare and submit the hypersurface fits for each truth point
    '''

    #
    # Get config
    #

    # Get the default neutrino pipeline, if not specificed
    if pipeline_cfg is None :
        pipeline_cfg = "neutrino"

    # Grab the hypersurface config
    nominal_dataset, sys_datasets, hypersurface_fit_kw, label = get_hypersurface_config(pipeline_cfg=pipeline_cfg, param_patches=param_patches, keep_param_names=keep_param_names, fix_intercept=fix_intercept) 

    #TODO check pipeline is neutrinos

    # Report
    if isinstance(pipeline_cfg, str) :
        print(f"Pipeline : {pipeline_cfg}")


    #
    # Prepare for fits
    #

    # Make the output directory
    # It should not already exist
    assert not os.path.exists(output_dir), "Output directory already exists (must remove before generating new hypersurfaces) : %s" % output_dir
    make_dir(output_dir)

    # Prepare the fits
    num_jobs = prepare_interpolated_fit(
        nominal_dataset=nominal_dataset,
        sys_datasets=sys_datasets,
        fit_directory=output_dir,
        interpolation_param_spec=interpolation_param_spec,
        **hypersurface_fit_kw
    )


    #
    # Init cluster submission
    #

    cluster_dir = os.path.join(output_dir, "cluster")
    make_dir(cluster_dir)

    submitter = ClusterSubmitter( 
        run_locally=False,
        job_name="upgrade_fit_hypersurfaces", 
        submit_dir=cluster_dir, 
        output_dir=cluster_dir, 
        flush_factor=flush_factor, 
        memory=4000, 
        disk_space=1000, # MB 
        start_up_commands=[ "source %s" % SETUP_SCRIPT ],
    )


    #
    # Generate and submit jobs
    #

    # Path to this script
    this_script = os.path.realpath(__file__)

    # Loop over jobs
    for job_index in range(num_jobs) :

        # Build command to run
        command = "python %s _run_neutrino_grid_fit" % (this_script)
        command += " --fit-dir %s" % output_dir
        command += " --job-index %i" % job_index
        print(command)

        # Add to submitter
        submitter.add(command)

    # Submit
    submitter.submit()
    submitter.report()


def fit_simple(output_dir, pipeline_cfg, tag, param_patches=None, kde=False, gaussian_filter=False, keep_param_names=None, fix_intercept=False) :
    '''
    Simple hypersurface fitting (e.g. no interpolation)
    '''

    # Optionally use KDE
    if kde :
        pipeline_cfg = switch_pipeline_to_kde(pipeline_cfg)

    # Grab the hypersurface config
    nominal_dataset, sys_datasets, hypersurface_fit_kw, label = get_hypersurface_config(
        pipeline_cfg=pipeline_cfg, 
        param_patches=param_patches, 
        keep_param_names=keep_param_names, # Can select only a subset of params for testing, but in general don't want to do this
        fix_intercept=fix_intercept,
    )
        
    # Make the output directory, if required
    if not os.path.exists(output_dir) :
        make_dir(output_dir)

    # Optionally use gaussian filter
    if gaussian_filter :
        if hypersurface_fit_kw is None :
            hypersurface_fit_kw = {}
        hypersurface_fit_kw["smooth_method"] = "gaussian_filter"
        #TODO smooth args (for now just use defaults)
        tag += "__gaussian_filter"

    # Perform fit (also saves to file)
    fit_results_file = fit_hypersurfaces(
        nominal_dataset=nominal_dataset,
        sys_datasets=sys_datasets,
        output_dir=output_dir,
        tag=tag,
        **hypersurface_fit_kw
    )

    return fit_results_file


def fit_neutrino_simple(output_dir, pipeline_cfg=None, **fit_simple_kw) :
    '''
    Fit neutrino hypersurfaces, with just a single fit at the nominal value of the pipleine params.
    Use this if you DON'T want interpolated hypersurfaces.
    '''

    # Get the neutrino pipeline
    if pipeline_cfg is None :
        pipeline_cfg = "neutrino"

     # Do the fit
    fit_results_file = fit_simple(
        output_dir=output_dir,
        pipeline_cfg=pipeline_cfg,
        **fit_simple_kw
    )

    return fit_results_file



if __name__ == "__main__" :

    from utils.filesys_tools import replace_file_ext
    from utils.script_tools import ScriptWrapper
    with ScriptWrapper(log_file=replace_file_ext(__file__,".log")) as script :

        #
        # Get args
        #

        from utils.argparse_tools import EnhancedArgumentParser
        parser = EnhancedArgumentParser(subparser_names=["submit_neutrino_grid_fits", "assemble_neutrino_grid_fits","fit_neutrino_simple", "_run_neutrino_grid_fit"])
        parser.add_argument(['submit_neutrino_grid_fits', 'assemble_neutrino_grid_fits', 'fit_neutrino_simple'], '-an','--analysis-name', type=str, required=False, default="numu_disappearance", help='Name of analysis to use' )
        parser.add_argument(['submit_neutrino_grid_fits', 'assemble_neutrino_grid_fits', 'fit_neutrino_simple'], '-mo','--mass-ordering', type=str, required=False, default='nh', help='Choose mass ordering' )
        parser.add_argument(['submit_neutrino_grid_fits', 'fit_neutrino_simple'], "--pipeline", type=str, required=False, default=None, help='Can optionally provide an pipeline.' )
        parser.add_argument(['submit_neutrino_grid_fits', 'fit_neutrino_simple'], "--fix-intercept", action="store_true", help='Can optionally fix the hypersurface intercepts.' )
        parser.add_argument('_run_neutrino_grid_fit', '--fit-dir', type=str, required=True, help='Fit directory path' )
        parser.add_argument('_run_neutrino_grid_fit', '--job-index', type=int, required=True, help='Fit job index' )
        parser.add_argument(['fit_neutrino_simple'], "-od", "--output-dir", type=str, required=True, help='Must provide an output directory path' )
        parser.add_argument(['fit_neutrino_simple'], '--tag', type=str, required=False, default=None, help='Can optionally provide a tag for the file name' )
        parser.add_argument(['fit_neutrino_simple'], "--kde", action="store_true", help='Can optionally use a KDE instead of histogram for the input pipelines' )
        parser.add_argument(['fit_neutrino_simple'], "--gaussian-filter", action="store_true", help='Can optionally use a Gaussian filter to smooth the the input hists' )
        parser.add_argument(['submit_neutrino_grid_fits', 'fit_neutrino_simple'], '--keep-param-names', type=str, nargs="+", required=False, default=None, help='Can optionally specific a subset of hypersurface params to consider' )
        args = parser.parse_args()


        #
        # Neutrino fit commands
        #
        
        if args.which == "submit_neutrino_grid_fits" :

            # Default pipeline
            if args.pipeline is None :
                args.pipeline = "neutrino"

            # Configure analysis
            _, pipeline_cfg, _, param_patches, _ = configure_analysis(
                analysis_name=args.analysis_name,
                template_settings=args.pipeline,
                real_data=False,
                mass_ordering=args.mass_ordering,
            )

            # Get the hypersurface grid fit steering
            hypersurface_config = ANALYSES[args.analysis_name]["hypersurface_config"]["neutrino"][args.mass_ordering]
            fit_dir = hypersurface_config["fit_directory"]
            interpolation_param_spec = hypersurface_config["interpolation_param_spec"]
            flush_factor = hypersurface_config.get("flush_factor", 1)

            # Prepare and submit a set of hypersurface fits for a gid of true pipeline param values
            # (typically true values of the physics parameter)
            submit_neutrino_grid_fits(
                pipeline_cfg=pipeline_cfg,
                param_patches=param_patches,
                output_dir=fit_dir,
                interpolation_param_spec=interpolation_param_spec,
                flush_factor=flush_factor,
                keep_param_names=args.keep_param_names,
                fix_intercept=args.fix_intercept,
            )


        elif args.which == "_run_neutrino_grid_fit" : # The leading underscore is to indicate that a user shouldn't directly run this command (it is "private", not user-facing)

            # Run a hypersurface fit for a single point in the submitted grid
            # A user does not directly call this, instead it is called by each job submitted via `submit_neutrino_grid_fits`
            run_interpolated_fit(fit_directory=args.fit_dir, job_idx=args.job_index, skip_successful=False)


        elif args.which == "assemble_neutrino_grid_fits" :

            # Get the paths
            hypersurface_config = ANALYSES[args.analysis_name]["hypersurface_config"]["neutrino"][args.mass_ordering]
            fit_dir = hypersurface_config["fit_directory"]
            assembled_file = hypersurface_config["assembled_file"]

            # Assemble the hypersurface fits for each grid point, e.g. the results of `submit_neutrino_grid_fits` once all the jobs are complete
            # Choosing the drop the stored fit map input data (not used in general, only there for data provenance) to keep the file size 
            # manageable (these are the signle biggest contributor).
            assemble_interpolated_fits(fit_directory=fit_dir, output_file=assembled_file, drop_fit_maps=True)


        elif args.which == "fit_neutrino_simple" :

            # Default pipeline
            if args.pipeline is None :
                args.pipeline = "neutrino"

            # Configure analysis
            _, pipeline_cfg, _, param_patches, _ = configure_analysis(
                analysis_name=args.analysis_name,
                template_settings=args.pipeline,
                mass_ordering=args.mass_ordering,
            )

            # Get the hypersurface grid fit steering
            if args.output_dir is None :
                args.output_dir = HYPERSURFACE_SETTINGS_PATH
            if args.tag is None :
                args.tag = get_default_tag(analysis_name=args.analysis_name, label="neutrino")

            # Perform a simple fit of the neutrino hypersurface (no grid itts, interpolation, etc, just a single fit at the nomnal pipeline param values)
            fit_neutrino_simple(
                pipeline_cfg=pipeline_cfg,
                param_patches=param_patches,
                output_dir=args.output_dir,
                tag=args.tag,
                kde=args.kde,
                gaussian_filter=args.gaussian_filter,
                keep_param_names=args.keep_param_names,
                fix_intercept=args.fix_intercept,
            )


        else :
            raise Exception("Unknown command : %s" % args.which)
