import os, collections, copy
import numpy as np

from pisa import ureg

from utils.container_tools import is_sequence

os.environ['PISA_RESOURCES'] = "$FRIDGE_DIR/analysis/:$FRIDGE_DIR/analysis/upgrade_std_osc/"

# Define setup script
SETUP_SCRIPT = os.path.join( os.path.dirname(__file__), "..", "..", "setup_analysis.sh" ) 

#
# Hypersurface steering
#

HYPERSURFACE_SETTINGS_PATH = '/data/sim/IceCubeUpgrade/pisa/level4_queso_v00_hypersurfaces/'

# Define hypersurface interpolation dimensions/values
INTERPOLATED_HYPERSURFACE_THETA23_VALUES = np.concatenate([ np.linspace(1., 30., num=3, endpoint=True), np.linspace(35., 55., num=11, endpoint=True), np.linspace(60, 89., num=3, endpoint=True) ]) * ureg["degree"] # # Finer spacing near maximal

INTERPOLATED_HYPERSURFACE_DELTAM31_VALUES_NH = np.concatenate([ [1.1], np.linspace(2., 3., num=11, endpoint=True), [3.9] ]) * 1e-3*ureg["eV**2"] # This is for the normal ordering. Fine spacing in the main region of interest.
INTERPOLATED_HYPERSURFACE_DELTAM31_VALUES_IH = np.sort( -1 * INTERPOLATED_HYPERSURFACE_DELTAM31_VALUES_NH ) # Use -ve version of normal ordering for inverted ordering

INTERPOLATED_HYPERSURFACE_DELTA_INDEX_VALUES = [-0.49, -0.2, -0.1, 0., +0.1, +0.2, +0.49] * ureg["dimensionless"] # Spans [-5 sigma, +5 sigma], with finer node spacing in the main region of interest [-2 sigma, +2 sigma]

#
# Analyses
#

SAMPLE = "upgrade_queso"
DATA_LOADER_STAGE_KEY = ('data', 'simple_data_loader')

ANALYSES = collections.OrderedDict()

ANALYSES["numu_disappearance"] = collections.OrderedDict(
    physics_params = ["theta23","deltam31"],
    hypersurface_config = {
        "neutrino" : {
            # Grid fit (for interpolation)
            "nh" : {
                "interpolation_param_spec" : collections.OrderedDict(
                    theta23 = { "values" : list(INTERPOLATED_HYPERSURFACE_THETA23_VALUES), "scales_log": False, },
                    deltam31 = { "values" : list(INTERPOLATED_HYPERSURFACE_DELTAM31_VALUES_NH), "scales_log": False, },
                    delta_index = { "values" : list(INTERPOLATED_HYPERSURFACE_DELTA_INDEX_VALUES), "scales_log": False, },
                ),
                "fit_directory" : os.path.join(HYPERSURFACE_SETTINGS_PATH, "neutrino_hypersurfaces_deltam31_theta23_delta_index_nh_grid_fits"),
                "assembled_file" : os.path.join(HYPERSURFACE_SETTINGS_PATH, "neutrino_hypersurfaces_deltam31_theta23_delta_index_nh_grid_fits.pckl"), # Using pickle files, better performance than json.bz2. Still too large for GitHub, so putting in data dir
                "flush_factor" : 2, # Need to group some grid points to avoid having more jobs than some clusters allow
            },
            "ih" : {
                "interpolation_param_spec" : collections.OrderedDict(
                    theta23 = { "values" : list(INTERPOLATED_HYPERSURFACE_THETA23_VALUES), "scales_log": False, },
                    deltam31 = { "values" : list(INTERPOLATED_HYPERSURFACE_DELTAM31_VALUES_IH), "scales_log": False, },
                    delta_index = { "values" : list(INTERPOLATED_HYPERSURFACE_DELTA_INDEX_VALUES), "scales_log": False, },
                ),
                "fit_directory" : os.path.join(HYPERSURFACE_SETTINGS_PATH, "neutrino_hypersurfaces_deltam31_theta23_delta_index_ih_grid_fits"),
                "assembled_file" : os.path.join(HYPERSURFACE_SETTINGS_PATH, "neutrino_hypersurfaces_deltam31_theta23_delta_index_ih_grid_fits.pckl"),
                "flush_factor" : 2,
            },
        },
    },
)

#
# Pipelines
#

MASS_ORDERINGS = ["nh", "ih"]
DEFAULT_MASS_ORDERING = "nh"

SETTINGS_DIR = os.path.join( os.path.dirname(__file__), "..", "settings" ) 
NEUTRINO_PIPELINE_NH = os.path.join(SETTINGS_DIR, "pipeline", "pipeline_upgrade_neutrinos_std_osc_NO.cfg")
NEUTRINO_PIPELINE_IH = os.path.join(SETTINGS_DIR, "pipeline", "pipeline_upgrade_neutrinos_std_osc_IO.cfg")

MUON_PIPELINE = os.path.join(SETTINGS_DIR, "pipeline", "pipeline_upgrade_muons.cfg")
BACKGROUND_PIPELINES = [ MUON_PIPELINE ]

#
# Helper functions
#

def get_neutrino_pipeline(mass_ordering=None) :
    '''
    Get the neutrino pipeline for the specified mass ordering
    '''

    if mass_ordering is None :
        mass_ordering = DEFAULT_MASS_ORDERING

    assert mass_ordering in MASS_ORDERINGS

    return NEUTRINO_PIPELINE_NH if mass_ordering == "nh" else NEUTRINO_PIPELINE_IH


def expand_pipeline_shortcut(pipeline_cfg, mass_ordering=None) :
    '''
    Handle shortcuts to pipeline
    Allows user to specify "neutrino" or "muon" instead of a full math
    '''

    # Handle "neutrino" case
    if pipeline_cfg == "neutrino" :
        return get_neutrino_pipeline(mass_ordering)

    # Handle "muon" case
    elif pipeline_cfg == "muon" :
        return MUON_PIPELINE

    # Otherwise, nothing to expand
    return pipeline_cfg


def configure_analysis(
    analysis_name,
    template_settings=None,
    data_settings=None,
    real_data=False,
    analysis_dict=ANALYSES,
    mass_ordering=None,
) :
    '''
    Function to handle all the analysis-specific configuration

    Making into a function so other scripts (e.g. hypersurface fits) can use it
    '''

    # Get the dict specifying the requested analysis
    assert isinstance(analysis_dict, collections.abc.Mapping), "`analysis_dict` is not a dict"
    assert analysis_name in analysis_dict, "Unknown analysis : %s (choose from %s)" % (analysis_name, list(analysis_dict.keys()))
    this_analysis_dict = analysis_dict[analysis_name]

    # Get physics params
    physics_params = copy.deepcopy(this_analysis_dict["physics_params"])

    # If user didn't specify pipelines, add the standard signal and 
    # background pipelines for the chosen analysis (for the chosen
    # mass ordering)
    if template_settings is None :
        template_settings = [ get_neutrino_pipeline(mass_ordering) ]
        template_settings.extend(BACKGROUND_PIPELINES)

    # If user did specify pipleines, expand shortcuts if necessary
    else :
        # Handle single pipeline vs list of pipelines
        if is_sequence(template_settings) :
            template_settings = [ expand_pipeline_shortcut(p, mass_ordering=mass_ordering) for p in template_settings ]
        else :
            template_settings = expand_pipeline_shortcut(template_settings, mass_ordering=mass_ordering)

    # Grab the parameter patches
    # `param_patches` is used for both template and data in general, unless `data_patches` is available in which case that is used for data
    #TODO should `data_patches` also include `param_patches`?
    template_patches = copy.deepcopy(this_analysis_dict["param_patches"]) if "param_patches" in this_analysis_dict else None
    data_patches = copy.deepcopy(this_analysis_dict["data_patches"]) if "data_patches" in this_analysis_dict else copy.deepcopy(template_patches)

    return physics_params, template_settings, data_settings, template_patches, data_patches
