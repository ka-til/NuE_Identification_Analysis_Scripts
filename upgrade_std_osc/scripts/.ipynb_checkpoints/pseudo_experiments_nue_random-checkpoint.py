import os
os.environ['PISA_RESOURCES'] = "/data/user/akatil/electron_neutrino/for_real/PISA/wg-oscillations-fridge/analysis/upgrade_std_osc/"
os.environ['PISA_RESOURCES'] += os.pathsep + "/data/user/akatil/electron_neutrino/for_real/PISA/wg-oscillations-fridge/analysis/"
os.environ['PISA_RESOURCES'] += os.pathsep + "/data/ana/LE/"
os.environ['PISA_RESOURCES'] += os.pathsep + "/data/user/mliubarska/osc/pisa2_osc/pisa_examples/resources/"

import numpy as np
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
import pickle
import pisa
import copy
from scipy.special import erfcinv, erfc

from pisa.core.pipeline import Pipeline
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.detectors import Detectors
from pisa import FTYPE, ureg
from pisa.utils.fileio import from_file, to_file
from pisa.core.map import MapSet
from pisa.analysis.analysis import Analysis
from pisa.analysis.analysis import update_param_values, update_param_values_detector

params = {'legend.fontsize': 18,
          'figure.figsize': (9, 9*0.618),
          'axes.labelsize': 18,
          'axes.titlesize': 18,
          'xtick.labelsize': 18,
          'ytick.labelsize': 18}
plt.rcParams.update(params)

def calc_sens(TO, WO=None):
    
    if WO is None:
        return np.sqrt(TO)
    else:
        return np.sqrt(2)*(TO+WO)/np.sqrt(8*WO)

p1_nu_NO = Pipeline("settings/pipeline/pipeline_upgrade_neutrinos_std_osc_NO_nue_random.cfg")
p1_nu_IO = Pipeline("settings/pipeline/pipeline_upgrade_neutrinos_std_osc_IO_nue_random.cfg")

shared_params = list(p1_nu_NO.params.free.names) #+ list(p1_mu.params.free.names)
template_maker_NO = DistributionMaker(p1_nu_NO) #Detectors([p1_nu_NO, p1_mu, p2_nu_NO], shared_params=shared_params)
template_maker_IO = DistributionMaker(p1_nu_IO)

analysis = Analysis()
analysis.pprint = False

# global minimizer
nlopt_settings = {
    "method": "nlopt",
    "method_kwargs": {
        "algorithm": "NLOPT_GN_CRS2_LM",
        "ftol_abs": 1e-3,
        "ftol_rel": 1e-3,
        # other options that can be set here: 
        # xtol_abs, xtol_rel, stopval, maxeval, maxtime
        # after maxtime seconds, stop and return best result so far
        "maxtime": 500
    },
    "local_fit_kwargs": None  # no further nesting available
}

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

# complete fit routine
fit_combi = {
    "method": "staged",
    "method_kwargs": None,
    "local_fit_kwargs":[
        nlopt_settings,
        fit_octant,
    ]
}

fake_data_nh = template_maker_NO.get_outputs(return_sum=True)

best_fit_info_nh = analysis.fit_recursively(
            data_dist=fake_data_nh,
            hypo_maker=template_maker_IO,
            metric=["mod_chi2"],
            external_priors_penalty=None,
            **fit_octant
        )

fake_data_ih = template_maker_IO.get_outputs(return_sum=True)

best_fit_info_ih = analysis.fit_recursively(
            data_dist=fake_data_ih,
            hypo_maker=template_maker_NO,
            metric=["mod_chi2"],
            external_priors_penalty=None,
            **fit_octant
        )

nh_asi = best_fit_info_nh.metric_val
ih_asi = best_fit_info_ih.metric_val
sens = calc_sens(nh_asi, ih_asi)
sens_ind0 = calc_sens(best_fit_info_nh['detailed_metric_info'][0]['mod_chi2']['maps']['total']+sum(best_fit_info_nh['detailed_metric_info'][0]['mod_chi2']['priors']), 
                      best_fit_info_ih['detailed_metric_info'][0]['mod_chi2']['maps']['total']+sum(best_fit_info_ih['detailed_metric_info'][0]['mod_chi2']['priors']))
sens_ind1 = calc_sens(best_fit_info_nh['detailed_metric_info'][1]['mod_chi2']['maps']['total']+sum(best_fit_info_nh['detailed_metric_info'][1]['mod_chi2']['priors']), 
                      best_fit_info_ih['detailed_metric_info'][1]['mod_chi2']['maps']['total']+sum(best_fit_info_ih['detailed_metric_info'][1]['mod_chi2']['priors']))

nh_nh, nh_ih, ih_nh, ih_ih = [], [], [], []
nh_dist, ih_dist = [], []
sens_true_no, sens_true_ind0, sens_true_ind1 = [], [], []

for i in range(20):
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


    nh_nh.append(best_fit_info_nh_nh.metric_val)
    nh_ih.append(best_fit_info_nh_ih.metric_val)
    ih_nh.append(best_fit_info_ih_nh.metric_val)
    ih_ih.append(best_fit_info_ih_ih.metric_val)
    
    nh_dist.append(best_fit_info_nh_nh.metric_val - best_fit_info_ih_nh.metric_val)
    ih_dist.append(best_fit_info_nh_ih.metric_val - best_fit_info_ih_ih.metric_val)

    sens_true_NO.append(calc_sens(ih_nh, nh_ih))
    sens_true_ind0.append(calc_sens(best_fit_info_ih_nh['detailed_metric_info'][0]['mod_chi2']['maps']['total']+sum(best_fit_info_ih_nh['detailed_metric_info'][0]['mod_chi2']['priors']),
                                   best_fit_info_nh_ih['detailed_metric_info'][0]['mod_chi2']['maps']['total']+sum(best_fit_info_nh_ih['detailed_metric_info'][0]['mod_chi2']['priors'])))
    sens_true_ind1.append(calc_sens(best_fit_info_ih_nh['detailed_metric_info'][1]['mod_chi2']['maps']['total']+sum(best_fit_info_ih_nh['detailed_metric_info'][1]['mod_chi2']['priors']),
                                   best_fit_info_nh_ih['detailed_metric_info'][1]['mod_chi2']['maps']['total']+sum(best_fit_info_nh_ih['detailed_metric_info'][1]['mod_chi2']['priors'])))
    #sens_true_IO.append(calc_sens(nh_ih, ih_nh))

plt.hist(nh_nh, histtype='step', linewidth=3)
plt.xlabel('mod_chi2')
plt.ylabel('#pseudo experiments')
plt.title('Data NO and hypothesis NO')
plt.savefig('/data/user/akatil/electron_neutrino/for_real/PISA/pseudo_experiments_nh_nh_upgrade_nue_random.png', bbox_inches='tight')
plt.clf()

plt.hist(nh_ih, histtype='step', linewidth=3)
plt.xlabel('mod_chi2')
plt.ylabel('#pseudo experiments')
plt.title('Data IO and hypothesis NO')
plt.savefig('/data/user/akatil/electron_neutrino/for_real/PISA/pseudo_experiments_nh_ih_upgrade_nue_random.png', bbox_inches='tight')
plt.clf()

plt.hist(ih_nh, histtype='step', linewidth=3)
plt.xlabel('mod_chi2')
plt.ylabel('#pseudo experiments')
plt.title('Data NO and hypothesis IO')
plt.savefig('/data/user/akatil/electron_neutrino/for_real/PISA/pseudo_experiments_ih_nh_upgrade_nue_random.png', bbox_inches='tight')
plt.clf()

plt.hist(ih_ih, histtype='step', linewidth=3)
plt.xlabel('mod_chi2')
plt.ylabel('#pseudo experiments')
plt.title('Data IO and hypothesis IO')
plt.savefig('/data/user/akatil/electron_neutrino/for_real/PISA/pseudo_experiments_ih_ih_upgrade_nue_random.png', bbox_inches='tight')
plt.clf()

plt.hist(sens_true_NO, histtype='step', linewidth=3)
plt.axvline(sens, color='tab:orange', label=r'Nominal Sensitivity')
plt.legend()
plt.xlabel('median sensitivity 0')
plt.ylabel('#pseudo experiments')
plt.title('True Ordering = NO')
plt.savefig('/data/user/akatil/electron_neutrino/for_real/PISA/pseudo_experiments_sensitivity_combined_upgrade_nue_random.png', bbox_inches='tight')
plt.clf()

plt.hist(sens_true_ind0, histtype='step', linewidth=3)
plt.axvline(sens_ind0, color='tab:orange', label=r'Nominal Sensitivity')
plt.legend()
plt.xlabel('median sensitivity 1')
plt.ylabel('#pseudo experiments')
plt.title('True Ordering = NO')
plt.savefig('/data/user/akatil/electron_neutrino/for_real/PISA/pseudo_experiments_sensitivity_0_upgrade_nue_random.png', bbox_inches='tight')
plt.clf()

plt.hist(sens_true_ind1, histtype='step', linewidth=3)
plt.axvline(sens_ind1, color='tab:orange', label=r'Nominal Sensitivity')
plt.legend()
plt.xlabel('median sensitivity')
plt.ylabel('#pseudo experiments')
plt.title('True Ordering = NO')
plt.savefig('/data/user/akatil/electron_neutrino/for_real/PISA/pseudo_experiments_sensitivity_1_upgrade_nue_random.png', bbox_inches='tight')
plt.clf()

'''
plt.hist(sens_true_IO, histtype='step', linewidth=3)
plt.xlabel('median sensitivity')
plt.ylabel('#pseudo experiments')
plt.title('True Ordering = IO')
'''

plt.hist(nh_dist, label=r'Pseudo$_{NO}$', histtype='step', linewidth=3)
plt.hist(ih_dist, label=r'Pseudo$_{IO}$', histtype='step', linewidth=3)
plt.axvline(-nh_asi, label=r'Asimov$_{NO}$')
plt.axvline(ih_asi, color='tab:orange', label=r'Asimov$_{IO}$')
plt.legend()
plt.xlabel(r'$\chi^{2}_{NO}-\chi^{2}_{IO}$')
plt.ylabel('#pseudo experiments')
plt.savefig('/data/user/akatil/electron_neutrino/for_real/PISA/pseudo_experiments_del_chi2_upgrade_nue_random.png', bbox_inches='tight')
plt.clf()