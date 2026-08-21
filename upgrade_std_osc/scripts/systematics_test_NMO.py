import os
os.environ['PISA_RESOURCES'] = "/data/user/jweldert/wg-oscillations-fridge/analysis/upgrade_std_osc/"
os.environ['PISA_RESOURCES'] += os.pathsep + "/data/user/jweldert/wg-oscillations-fridge/analysis/"
os.environ['PISA_RESOURCES'] += os.pathsep + "/data/ana/LE/"
os.environ['PISA_RESOURCES'] += os.pathsep + "/data/user/jweldert/pisa/pisa_examples/resources/"

import math
import pickle
import numpy as np
from uncertainties import unumpy as unp
from optparse import OptionParser

import pisa
from pisa.analysis.analysis import Analysis
from pisa.core.pipeline import Pipeline
from pisa.core.distribution_maker import DistributionMaker
from pisa import FTYPE, ureg
from pisa.utils.fileio import to_file, from_file
from pisa.analysis.analysis import update_param_values


usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("--mo", default='NO', dest="MO", help="True Mass Ordering")
parser.add_option("--ref", action='store_true', dest="REF", help="Do the reference fit")
parser.add_option("--p", type=str, default=None, dest="P", help="Parameter to be tested")
parser.add_option("--i", type=int, default=None, dest="I", help="Index of parameter to be tested")
parser.add_option("--allfree", action='store_true', dest="FREE", help="Leave all params free for the fits")
(options,args) = parser.parse_args()

MO, REF, P, I, FREE = options.MO, options.REF, options.P, options.I, options.FREE
assert MO in ['NO', 'IO'], "Choose either NO or IO for mass ordering"
assert (P is None) + (I is None) == 1, "Specify either parameter name or index"
folder = "/data/user/jweldert/wg-oscillations-fridge/analysis/upgrade_std_osc/scans/NMO/systematics/"


template_maker_NO = DistributionMaker(["settings/pipeline/pipeline_upgrade_neutrinos_std_osc_NO.cfg", 
                                       "settings/pipeline/pipeline_upgrade_muons.cfg"])
template_maker_IO = DistributionMaker(["settings/pipeline/pipeline_upgrade_neutrinos_std_osc_IO.cfg", 
                                       "settings/pipeline/pipeline_upgrade_muons.cfg"])


# all parameters of interest (first free, then fixed)
ALL_PARAMS = ['delta_index', 'barr_d_Pi', 'barr_g_Pi', 'barr_h_Pi', 'barr_i_Pi', 'barr_w_K', 'barr_z_K', 'theta23',
              'deltam31', 'theta13', 'Genie_Ma_QE', 'Genie_Ma_RES', 'Genie_Ma_RES_NC', 'Genie_Ma_COH_PI',
              'nutau_xsec_scale', 'aeff_scale', 'dom_eff', 'icu_dom_eff', 'bulk_ice_abs', 'bulk_ice_scatter', #
              'barr_a_Pi', 'barr_b_Pi', 'barr_c_Pi', 'barr_e_Pi', 'barr_f_Pi', 'barr_x_K', 'barr_y_K',
              'barr_w_antiK', 'barr_x_antiK', 'barr_y_antiK', 'barr_z_antiK', 'theta12', 'deltacp', 'deltam21',
              'dis_csms'
             ]
if FREE:
    folder += "allFree/"
    for p in ALL_PARAMS:
        template_maker_NO.params[p].is_fixed = False
        template_maker_IO.params[p].is_fixed = False

if P is None:
    P = ALL_PARAMS[I]
else:
    assert P in ALL_PARAMS


analysis = Analysis()
analysis.pprint = False

# local minimizer
local_fit_minuit = {
    "method": "iminuit",
    "method_kwargs": {
        "errors": 0.1,
        "precision": 1e-14,  # default: double precision
        "tol": 1e-2,  # default: 0.1
        "run_simplex": False,
        "run_migrad": True
    },
    "local_fit_kwargs": None
}

# octant fit for local minimizer
fit_octant = {
    "method": "octants",
    "method_kwargs": {
        "angle": "theta23",
        "inflection_point": 45 * ureg.degrees,
    },
    "local_fit_kwargs": local_fit_minuit
}


if REF: # Reference fit
    if MO == 'NO':
        fake_data_NO = template_maker_NO.get_outputs(return_sum=True)

        best_fit_info_ref = analysis.fit_recursively(
                                data_dist=fake_data_NO,
                                hypo_maker=template_maker_IO,
                                metric=["mod_chi2"],
                                external_priors_penalty=None,
                                **fit_octant
                            )
        template_maker_IO.params.reset_free()
    else:
        fake_data_IO = template_maker_IO.get_outputs(return_sum=True)

        best_fit_info_ref = analysis.fit_recursively(
                                data_dist=fake_data_IO,
                                hypo_maker=template_maker_NO,
                                metric=["mod_chi2"],
                                external_priors_penalty=None,
                                **fit_octant
                            )
        template_maker_NO.params.reset_free()

    to_file(best_fit_info_ref, folder+"reference_fit_%s.json"%(MO))
#else:
#    best_fit_info_ref = from_file(folder+"reference_fit_%s.json"%(MO))


# Fitting Relevance Test
results = {}

if MO == 'NO':
    if P == 'theta23':
        template_maker_NO.params[P].value -= 1.2 * ureg.degree
    elif template_maker_NO.params[P].prior.kind == 'uniform':
        template_maker_NO.params[P] = template_maker_NO.params[P].range[0]
    elif template_maker_NO.params[P].prior.kind == 'gaussian':
        template_maker_NO.params[P].value -= template_maker_NO.params[P].prior.stddev
    else:
        raise ValueError("Parameter prior is of kind %s which is not supported"%(template_maker_NO.params[P].prior.kind))
    fake_data_NO = template_maker_NO.get_outputs(return_sum=True)

    template_maker_NO.params[P] = template_maker_NO.params[P].nominal_value
    template_maker_NO.params[P].is_fixed = True
    template_maker_IO.params[P].is_fixed = True

    best_fit_info_ih = analysis.fit_recursively(
                        data_dist=fake_data_NO,
                        hypo_maker=template_maker_IO,
                        metric=["mod_chi2"],
                        external_priors_penalty=None,
                        **fit_octant
                    )

    best_fit_info_nh = analysis.fit_recursively(
                        data_dist=fake_data_NO,
                        hypo_maker=template_maker_NO,
                        metric=["mod_chi2"],
                        external_priors_penalty=None,
                        **fit_octant
                    )

    results['neg'] = np.array([best_fit_info_ih['metric_val'], best_fit_info_nh['metric_val']])


    template_maker_NO.params.reset_free()
    if P == 'theta23':
        template_maker_NO.params[P].value += 1.2 * ureg.degree
    elif template_maker_NO.params[P].prior.kind == 'uniform':
        template_maker_NO.params[P] = template_maker_NO.params[P].range[1]
    elif template_maker_NO.params[P].prior.kind == 'gaussian':
        template_maker_NO.params[P].value += template_maker_NO.params[P].prior.stddev
    else:
        raise ValueError("Parameter prior is of kind %s which is not supported"%(template_maker_NO.params[P].prior.kind))
    fake_data_NO = template_maker_NO.get_outputs(return_sum=True)

    template_maker_NO.params[P] = template_maker_NO.params[P].nominal_value

    best_fit_info_ih = analysis.fit_recursively(
                        data_dist=fake_data_NO,
                        hypo_maker=template_maker_IO,
                        metric=["mod_chi2"],
                        external_priors_penalty=None,
                        **fit_octant
                    )

    best_fit_info_nh = analysis.fit_recursively(
                        data_dist=fake_data_NO,
                        hypo_maker=template_maker_NO,
                        metric=["mod_chi2"],
                        external_priors_penalty=None,
                        **fit_octant
                    )

    results['pos'] = np.array([best_fit_info_ih['metric_val'], best_fit_info_nh['metric_val']])
    
else:
    if P == 'theta23':
        template_maker_IO.params[P].value -= 1.2 * ureg.degree
    elif template_maker_IO.params[P].prior.kind == 'uniform':
        template_maker_IO.params[P] = template_maker_IO.params[P].range[0]
    elif template_maker_IO.params[P].prior.kind == 'gaussian':
        template_maker_IO.params[P].value -= template_maker_IO.params[P].prior.stddev
    else:
        raise ValueError("Parameter prior is of kind %s which is not supported"%(template_maker_IO.params[P].prior.kind))
    fake_data_IO = template_maker_IO.get_outputs(return_sum=True)

    template_maker_IO.params[P] = template_maker_IO.params[P].nominal_value
    template_maker_NO.params[P].is_fixed = True
    template_maker_IO.params[P].is_fixed = True

    best_fit_info_ih = analysis.fit_recursively(
                        data_dist=fake_data_IO,
                        hypo_maker=template_maker_IO,
                        metric=["mod_chi2"],
                        external_priors_penalty=None,
                        **fit_octant
                    )

    best_fit_info_nh = analysis.fit_recursively(
                        data_dist=fake_data_IO,
                        hypo_maker=template_maker_NO,
                        metric=["mod_chi2"],
                        external_priors_penalty=None,
                        **fit_octant
                    )

    results['neg'] = np.array([best_fit_info_ih['metric_val'], best_fit_info_nh['metric_val']])


    template_maker_IO.params.reset_free()
    if P == 'theta23':
        template_maker_IO.params[P].value += 1.2 * ureg.degree
    elif template_maker_IO.params[P].prior.kind == 'uniform':
        template_maker_IO.params[P] = template_maker_IO.params[P].range[1]
    elif template_maker_IO.params[P].prior.kind == 'gaussian':
        template_maker_IO.params[P].value += template_maker_IO.params[P].prior.stddev
    else:
        raise ValueError("Parameter prior is of kind %s which is not supported"%(template_maker_IO.params[P].prior.kind))
    fake_data_IO = template_maker_IO.get_outputs(return_sum=True)

    template_maker_IO.params[P] = template_maker_IO.params[P].nominal_value

    best_fit_info_ih = analysis.fit_recursively(
                        data_dist=fake_data_IO,
                        hypo_maker=template_maker_IO,
                        metric=["mod_chi2"],
                        external_priors_penalty=None,
                        **fit_octant
                    )

    best_fit_info_nh = analysis.fit_recursively(
                        data_dist=fake_data_IO,
                        hypo_maker=template_maker_NO,
                        metric=["mod_chi2"],
                        external_priors_penalty=None,
                        **fit_octant
                    )

    results['pos'] = np.array([best_fit_info_ih['metric_val'], best_fit_info_nh['metric_val']])


to_file(results, folder+P+"_fit_%s.json"%(MO))
