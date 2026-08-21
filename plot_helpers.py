import numpy as np
import matplotlib.pyplot as plt


def mctree_plot(CC, NC, mask_data, min_val, max_val, xlabel, 
                energy=False, charge=False, charge_mdom=False,
                bins=100, histtype='step', density=True, lw=2, log=False):

    """
    Takes
    CC and NC arugument
    mask data contains CC and NC energy, total_charge, total_charge_m
    min_val and max_val can be either energy or charge
    xlabel should be chosen based on the argumen being passed
    energy, charge and charge_mdom are boolean, only one of these can be true.
    bins, histogram, density and lw are histogram parameters
    log is a boolean, if true, np.log10 is applied to the CC and NC arguments are 
    """

    if energy:
        #CC and NC arguments are grouped based on energy
        masking_propCC = mask_data[0]
        masking_propNC = mask_data[1]
        title = 'Energy[GeV]'

    if charge:
        #CC and NC arguments are grouped based on total charge
        masking_propCC = mask_data[2]
        masking_propNC = mask_data[3]
        title = 'Charge'

    if charge_mdom:
        #CC and NC arguments are grouped based on total mdom charge
        masking_propCC = mask_data[4]
        masking_propNC = mask_data[5]
        title = 'mDOM Charge'
        
    maskCC = (masking_propCC >= min_val)&(masking_propCC < max_val)
    maskNC = (masking_propNC >= min_val)&(masking_propNC < max_val)

    if log==True:
        CC = np.log10(CC[maskCC])
        NC = np.log10(NC[maskNC])
    else:
        CC = CC[maskCC]
        NC = NC[maskNC]
    
    plt.hist(CC[np.isfinite(CC)], bins=bins, histtype=histtype, density=density, lw=lw, label='CC')
    plt.hist(NC[np.isfinite(NC)], bins=bins, histtype=histtype, density=density, lw=lw, label='NC')
    #plt.hist(CC[np.isfinite(CC)], bins=bins, histtype=histtype, density=density, lw=lw, label=r'$\nu_e$')
    #plt.hist(NC[np.isfinite(NC)], bins=bins, histtype=histtype, density=density, lw=lw, label=r'$\bar{\nu}_e$')
    plt.legend()
    plt.xlabel(xlabel)
    plt.title('{} {} - {}'.format(title, min_val, max_val))
    plt.show()
