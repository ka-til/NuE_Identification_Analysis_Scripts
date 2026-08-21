import os
os.environ['PISA_RESOURCES'] = "/data/user/akatil/electron_neutrino/for_real/PISA/wg-oscillations-fridge/analysis/upgrade_std_osc/"
os.environ['PISA_RESOURCES'] += os.pathsep + "/data/user/akatil/electron_neutrino/for_real/PISA/wg-oscillations-fridge/analysis/"
os.environ['PISA_RESOURCES'] += os.pathsep + "/data/ana/LE/"
os.environ['PISA_RESOURCES'] += os.pathsep + "/data/user/mliubarska/osc/pisa2_osc/pisa_examples/resources/"#"/data/user/jweldert/pisa/pisa_examples/resources/"

import math
import pickle
import numpy as np
from uncertainties import unumpy as unp
from optparse import OptionParser

import pisa
from pisa.analysis.analysis import Analysis
from pisa.core.pipeline import Pipeline
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.detectors import Detectors
from pisa import FTYPE, ureg
from pisa.utils.fileio import to_file
from pisa.analysis.analysis import update_param_values, update_param_values_detector


usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("--dc", action='store_true', dest="DC", help="DC only")
parser.add_option("--pe", type=int, default=0, dest="PE", help="Pseudo Experiments")
parser.add_option("--lt", type=float, default=None, dest="LT", help="Livetime")
parser.add_option("--t23", type=float, default=None, dest="T23", help="True theta23 value (degree)")
parser.add_option("--mo", default='IO', dest="MO", help="True Mass Ordering")
parser.add_option("--juno", action='store_true', dest="JUNO", help="Combine with JUNO")
parser.add_option("--sig", default='nominal', dest="SIG", help="Sensitivity within Nufit 5.2 theta23 3 sigma range")
parser.add_option("--old", action='store_true', default=False, dest="OLD", help="Uses old (28) simulation pipeline")
(options,args) = parser.parse_args()

DC, PE, LT, T23, MO, JUNO, SIG, OLD = options.DC, options.PE, options.LT, options.T23, options.MO, options.JUNO, options.SIG, options.OLD
if not T23 is None:
    assert T23 > 0 and T23 < 90, "Theta23 must be between 0 and 90 degree"
    assert SIG == 'nominal', "theta23 scan will overwrite true value"
assert MO in ['NO', 'IO'], "Choose either NO or IO for mass ordering"
assert not (DC and JUNO), "Only DC and JUNO not supported"
assert SIG in ['low', 'nominal', 'high'], "sig must be one of 'low', 'nominal', 'high'"


if DC:
    name = 'dc_NMO_'
elif JUNO:
    name = 'juno_combi_NMO_'
else:
    name = 'upgrade_NMO_only_upgrade_no_masking_'
folder = '/data/user/akatil/electron_neutrino/for_real/PISA/wg-oscillations-fridge/analysis/upgrade_std_osc/scans/NMO/'#'/data/user/jweldert/wg-oscillations-fridge/analysis/upgrade_std_osc/scans/NMO/'
if OLD:
    folder += 'old/'
    old = '_old'
else:
    old = ''
if not T23 is None:
    folder += 'theta23_scan/'
    end = "_t23_%.1f.json"%(T23)
elif SIG == 'nominal':
    folder += 'livetime/'
    end = '.json'
else:
    folder += 'livetime_'+SIG+'/'
    end = '.json'


if DC:
    NuFit_t23_NO = {'low': 41.5,
                    'high': 48.9,
                   }
    NuFit_t23_IO = {'low': 48.5,
                    'high': 43.3,
                   }
else:
    NuFit_t23_NO = {'low': 41.8,
                    'high': 49.5,
                   }
    NuFit_t23_IO = {'low': 48.2,
                    'high': 43.7,
                   }


PARAMS_FREE = ['delta_index', 'barr_d_Pi', 'barr_g_Pi', 'barr_h_Pi', 'barr_i_Pi', 'barr_w_K', 'barr_z_K', 'theta23',
               'deltam31', 'theta13', 'Genie_Ma_QE', 'Genie_Ma_RES', 'Genie_Ma_RES_NC', 'Genie_Ma_COH_PI',
               'nutau_xsec_scale', 'aeff_scale', 'dom_eff', 'icu_dom_eff', 'bulk_ice_abs', 'bulk_ice_scatter', 
               'hole_ice_p0', 'hole_ice_p1', 'weight_scale']
def free_params(pipeline):
    for p in pipeline.params.names:
        if p in PARAMS_FREE:
            pipeline.params[p].is_fixed = False
        else:
            pipeline.params[p].is_fixed = True


if DC:
    template_maker_NO = DistributionMaker(["settings/pipeline/pipeline_oscnext_neutrinos_std_osc_NO.cfg"])
    template_maker_IO = DistributionMaker(["settings/pipeline/pipeline_oscnext_neutrinos_std_osc_IO.cfg"])
    
    free_params(template_maker_NO)
    free_params(template_maker_IO)
    update_param_values(template_maker_NO, template_maker_NO.params, update_is_fixed=True)
    update_param_values(template_maker_IO, template_maker_IO.params, update_is_fixed=True)
        
    if not LT is None:
        template_maker_NO.params.livetime = LT * ureg.common_year
        template_maker_IO.params.livetime = LT * ureg.common_year
    else:
        LT = template_maker_NO.params.livetime.value.magnitude
        assert LT == template_maker_IO.params.livetime.value.magnitude
        
    if not T23 is None:
        template_maker_NO.params.theta23 = T23 * ureg.degrees
        template_maker_IO.params.theta23 = T23 * ureg.degrees
    elif SIG in ['low', 'high']:
        template_maker_NO.params.theta23 = NuFit_t23_NO[SIG] * ureg.degrees
        template_maker_IO.params.theta23 = NuFit_t23_IO[SIG] * ureg.degrees

else:
    p1_nu_NO = Pipeline("settings/pipeline/pipeline_upgrade_neutrinos_std_osc_NO%s.cfg"%(old))
    free_params(p1_nu_NO)
    p1_nu_IO = Pipeline("settings/pipeline/pipeline_upgrade_neutrinos_std_osc_IO%s.cfg"%(old))
    free_params(p1_nu_IO)
    #p1_mu = Pipeline("settings/pipeline/pipeline_upgrade_muons%s.cfg"%(old))
    #p2_nu_NO = Pipeline("settings/pipeline/pipeline_oscnext_neutrinos_std_osc_NO.cfg")
    #free_params(p2_nu_NO)
    #p2_nu_IO = Pipeline("settings/pipeline/pipeline_oscnext_neutrinos_std_osc_IO.cfg")
    #free_params(p2_nu_IO)
    #p2_mu = Pipeline("settings/pipeline/pipeline_oscnext_muons.cfg")
    if JUNO:
        p3_nu_NO = Pipeline("settings/pipeline/pipeline_juno_NO.cfg")
        p3_nu_IO = Pipeline("settings/pipeline/pipeline_juno_IO.cfg")
        p3_bck = Pipeline("settings/pipeline/pipeline_juno_background.cfg")
    
    if not LT is None:
        p1_nu_NO.params.livetime = LT * ureg.common_year
        p1_nu_IO.params.livetime = LT * ureg.common_year
        #p1_mu.params.livetime = LT * ureg.common_year
        #p2_nu_NO.params.livetime = 12.0 * ureg.common_year        
        #p2_nu_IO.params.livetime = 12.0 * ureg.common_year
        if JUNO:
            p3_nu_NO.params.livetime = LT * 300 * ureg.day
            p3_nu_IO.params.livetime = LT * 300 * ureg.day
            p3_bck.params.livetime = LT * 300 * ureg.day
    else:
        LT = p1_nu_NO.params.livetime.value.magnitude
        assert LT == p1_nu_IO.params.livetime.value.magnitude
        
    if not T23 is None:
        p1_nu_NO.params.theta23 = T23 * ureg.degrees
        p1_nu_IO.params.theta23 = T23 * ureg.degrees
        #p2_nu_NO.params.theta23 = T23 * ureg.degrees
        #p2_nu_IO.params.theta23 = T23 * ureg.degrees
        if JUNO:
            p3_nu_NO.params.theta23 = T23 * ureg.degrees
            p3_nu_IO.params.theta23 = T23 * ureg.degrees
    elif SIG in ['low', 'high']:
        p1_nu_NO.params.theta23 = NuFit_t23_NO[SIG] * ureg.degrees
        p1_nu_IO.params.theta23 = NuFit_t23_IO[SIG] * ureg.degrees
        #p2_nu_NO.params.theta23 = NuFit_t23_NO[SIG] * ureg.degrees
        #p2_nu_IO.params.theta23 = NuFit_t23_IO[SIG] * ureg.degrees
        if JUNO:
            p3_nu_NO.params.theta23 = NuFit_t23_NO[SIG] * ureg.degrees
            p3_nu_IO.params.theta23 = NuFit_t23_IO[SIG] * ureg.degrees

    shared_params = list(p1_nu_NO.params.free.names) #+ list(p1_mu.params.free.names)
    #shared_params.remove('icu_dom_eff')
    if JUNO:
        shared_params.remove('aeff_scale')
        template_maker_NO = Detectors([p1_nu_NO, p1_mu, p2_nu_NO, p3_nu_NO, p3_bck], shared_params=shared_params)
        template_maker_IO = Detectors([p1_nu_IO, p1_mu, p2_nu_IO, p3_nu_IO, p3_bck], shared_params=shared_params)
    else:
        template_maker_NO = DistributionMaker(p1_nu_NO) #Detectors([p1_nu_NO, p1_mu, p2_nu_NO], shared_params=shared_params)
        template_maker_IO = DistributionMaker(p1_nu_IO) #Detectors([p1_nu_IO, p1_mu, p2_nu_IO], shared_params=shared_params)


if MO == 'NO' and os.path.exists(folder+name+'nh_nh_'+str(int(LT))+end):
    raise NameError("File %s already exists"%(folder+name+'nh_nh_'+str(int(LT))+end))
elif MO == 'IO' and os.path.exists(folder+name+'nh_ih_'+str(int(LT))+end):
    raise NameError("File %s already exists"%(folder+name+'nh_ih_'+str(int(LT))+end))


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


if JUNO:
    metrics = ["mod_chi2", "mod_chi2", "chi2"]
else:
    metrics = ["mod_chi2"]

if MO == 'NO':
    fake_data_nh = template_maker_NO.get_outputs(return_sum=True)

    best_fit_info_nh = analysis.fit_recursively(
                data_dist=fake_data_nh,
                hypo_maker=template_maker_IO,
                metric=metrics,
                external_priors_penalty=None,
                **fit_octant
            )

    fake_data_ih = template_maker_IO.get_outputs(return_sum=True)

    best_fit_info_ih = analysis.fit_recursively(
                data_dist=fake_data_ih,
                hypo_maker=template_maker_NO,
                metric=metrics,
                external_priors_penalty=None,
                **fit_octant
            )
else:
    fake_data_ih = template_maker_IO.get_outputs(return_sum=True)

    best_fit_info_ih = analysis.fit_recursively(
                data_dist=fake_data_ih,
                hypo_maker=template_maker_NO,
                metric=metrics,
                external_priors_penalty=None,
                **fit_octant
            )

    fake_data_nh = template_maker_NO.get_outputs(return_sum=True)

    best_fit_info_nh = analysis.fit_recursively(
                data_dist=fake_data_nh,
                hypo_maker=template_maker_IO,
                metric=metrics,
                external_priors_penalty=None,
                **fit_octant
            )

'''
# not saved so far
nh_dist, ih_dist = [], []
for i in range(PE):
    best_fit_info_nh_nh = analysis.fit_recursively(
                data_dist=fake_data_nh.fluctuate(method='poisson', random_state=i),
                hypo_maker=template_maker_NO,
                metric=["mod_chi2"],
                external_priors_penalty=None,
                **fit_octant
            )
    best_fit_info_nh_ih = analysis.fit_recursively(
                data_dist=fake_data_ih.fluctuate(method='poisson', random_state=i),
                hypo_maker=template_maker_NO,
                metric=["mod_chi2"],
                external_priors_penalty=None,
                **fit_octant
            )

    best_fit_info_ih_nh = analysis.fit_recursively(
                data_dist=fake_data_nh.fluctuate(method='poisson', random_state=i),
                hypo_maker=template_maker_IO,
                metric=["mod_chi2"],
                external_priors_penalty=None,
                **fit_octant
            )
    best_fit_info_ih_ih = analysis.fit_recursively(
                data_dist=fake_data_ih.fluctuate(method='poisson', random_state=i),
                hypo_maker=template_maker_IO,
                metric=["mod_chi2"],
                external_priors_penalty=None,
                **fit_octant
            )

    nh_dist.append(best_fit_info_nh_nh.metric_val - best_fit_info_ih_nh.metric_val)
    ih_dist.append(best_fit_info_nh_ih.metric_val - best_fit_info_ih_ih.metric_val)
'''

if not os.path.exists(folder):
    os.system("mkdir %s"%(folder))
if MO == 'NO':
    to_file(best_fit_info_nh, folder+name+'nh_nh_'+str(int(LT))+end)
    to_file(best_fit_info_ih, folder+name+'ih_nh_'+str(int(LT))+end)
else:
    to_file(best_fit_info_nh, folder+name+'nh_ih_'+str(int(LT))+end)
    to_file(best_fit_info_ih, folder+name+'ih_ih_'+str(int(LT))+end)
