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
from pisa.core.detectors import Detectors
from pisa import FTYPE, ureg
from pisa.utils.fileio import to_file
from pisa.analysis.analysis import update_param_values, update_param_values_detector


def get_value(name, v):
    if name == 'deltam31':
        return v * ureg.electron_volt**2
    elif name == 'theta23':
        return math.asin(math.sqrt(v)) * 180 / math.pi * ureg.degrees
    else:
        return v


usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("-p", "--parameter", action='append', dest="PARAMETER", help="parameter that should be changed")
parser.add_option("-v", "--value", action='append', dest="VALUE", help="parameter value")
parser.add_option("--dc", action='store_true', default=False, dest="DC", help="DC only")
parser.add_option("--icu", action='store_true', default=False, dest="ICU", help="ICU only")
parser.add_option("--old", action='store_true', default=False, dest="OLD", help="Uses old (28) simulation pipeline")
parser.add_option("--lt", type=float, default=None, dest="LT", help="Livetime")
(options,args) = parser.parse_args()

DC, ICU, OLD, LT = options.DC, options.ICU, options.OLD, options.LT
assert not (DC and ICU), "Can't fit DC only and ICU only"
parameters, values = options.PARAMETER, np.array(options.VALUE).astype(float)
assert len(parameters) == len(values), "Specify exactly one value for each parameter"
assert len(parameters) > 0, "No parameters provided"


if DC:
    name = 'dc_fit'
else:
    name = 'upgrade_fit'
for i, p in enumerate(parameters):
    name += '_' + p + '_%s'%(values[i])
name += '.json'

folder = '/data/user/jweldert/wg-oscillations-fridge/analysis/upgrade_std_osc/scans/'
if 'nutau_norm' in parameters:
    folder += 'nutau_appearance/'
else:
    folder += 'numu_disappearance/'
if OLD:
    folder += 'old/'
    old = '_old'
else:
    old = ''
if DC:
    folder += 'DC_only/'
elif ICU:
    folder += 'ICU_only/'
folder += str(int(LT)) + '_years/'

if os.path.exists(folder+name):
    raise NameError("File %s already exists"%(folder+name))


if DC:
    template_maker = DistributionMaker(["settings/pipeline/pipeline_oscnext_neutrinos_std_osc_NO.cfg"])
    
    if not LT is None:
        template_maker.params.livetime = LT * ureg.common_year
    else:
        LT = template_maker.params.livetime.value.magnitude
elif ICU:
    template_maker = DistributionMaker(["settings/pipeline/pipeline_upgrade_neutrinos_std_osc_NO%s.cfg"%(old),
                                        "settings/pipeline/pipeline_upgrade_muons%s.cfg"%(old)])
    
    if not LT is None:
        template_maker.params.livetime = LT * ureg.common_year
    else:
        LT = template_maker.params.livetime.value.magnitude
else:
    p1_nu = Pipeline("settings/pipeline/pipeline_upgrade_neutrinos_std_osc_NO%s.cfg"%(old))
    p1_mu = Pipeline("settings/pipeline/pipeline_upgrade_muons%s.cfg"%(old))
    p2_nu = Pipeline("settings/pipeline/pipeline_oscnext_neutrinos_std_osc_NO.cfg")
    #p2_mu = Pipeline("settings/pipeline/pipeline_oscnext_muons.cfg")
    
    for p in parameters:
        if p in p1_nu.params.names:
            p1_nu.params[p].is_fixed = True
        if p in p2_nu.params.names:
            p2_nu.params[p].is_fixed = True
            
    if not LT is None:
        p1_nu.params.livetime = LT * ureg.common_year
        p1_mu.params.livetime = LT * ureg.common_year
        p2_nu.params.livetime = 12.0 * ureg.common_year
    else:
        LT = p1_nu.params.livetime.value.magnitude

    shared_params = list(p1_nu.params.free.names) #+ list(p1_mu.params.free.names)
    shared_params.remove('icu_dom_eff')
    template_maker = Detectors([p1_nu, p1_mu, p2_nu], shared_params=shared_params)


for p in parameters:
    assert p in template_maker.params.names

template_maker.params.reset_free()
fake_data = template_maker.get_outputs(return_sum=True)


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


params = template_maker.params
for i, p in enumerate(parameters):
    params[p].value = get_value(p, values[i])
    params[p].is_fixed = True
    if p + '_deepcore' in params.names:
        params[p + '_deepcore'].value = get_value(p, values[i])
        params[p + '_deepcore'].is_fixed = True
if DC or ICU:
    update_param_values(template_maker, params)
else:
    update_param_values_detector(template_maker, params)

best_fit_info = analysis.fit_recursively(
            data_dist=fake_data,
            hypo_maker=template_maker,
            metric=["mod_chi2"],
            external_priors_penalty=None,
            **fit_octant
        )


if not os.path.exists(folder):
    os.system("mkdir %s"%(folder))
to_file(best_fit_info, folder+name)
