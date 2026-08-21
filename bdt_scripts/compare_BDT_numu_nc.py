import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib

params = {'figure.figsize': (7, 7*0.618),
          'legend.fontsize': 14,
          'axes.labelsize': 16,
          'axes.titlesize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16}
plt.rcParams.update(params)

df = pd.read_hdf('/data/user/akatil/electron_neutrino/for_real/dataset_complete/BDT/BDT_all_updated_with_muon.h5', key='data')

df.dropna(inplace=True)

data = df[(df['TrueEnergy']>=2)&(df['TrueEnergy']<=15)&(df['TrueZenith']<=-0.3)]
data = data.groupby('Label').sample(90000, random_state=0)

y = data.Label

features = ['RecoEnergy',
'TotalCharge',
'CogDistance',
 'Beta1Reco',
 'tres_median',
 'tres_iqr',
 'tres_skew',
 'dist_iqr',
 'dist_skew',
 'ang_median',
 'ang_iqr',
 'ang_kurtosis',
 'ang_skew']

X = data[features]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state=0)

model_nc = joblib.load('/data/user/akatil/electron_neutrino/for_real/analysis_chain/bdt_scripts/submit_scripts/seventh_iter.joblib')
model_numu = joblib.load('/data/user/akatil/electron_neutrino/for_real/analysis_chain/bdt_scripts/submit_scripts/numu_first_iter.joblib')

#ROC-AUC curve
prob_y_nc = model_nc.predict_proba(test_X)[:, 1]
prob_y_numu = model_numu.predict_proba(test_X)[:, 1]

prob_signal_numu = prob_y_numu[test_y == 1]
prob_background_numu = prob_y_numu[test_y == 2]
prob_background2_numu = prob_y_numu[test_y == 0]

test_y = test_y#[prob_y_numu >= 0.5]
prob_y_nc = prob_y_nc#[prob_y_numu >= 0.5]
prob_signal_nc = prob_y_nc[test_y == 1]
prob_background_nc = prob_y_nc[test_y == 0]
prob_background2_nc = prob_y_nc[test_y == 2]

bins_prob = np.arange(0, 1, 0.02)

#model 7
counts_signal, bin_edges = np.histogram(prob_signal_nc, bins=bins_prob)
counts_background, _ = np.histogram(prob_background_nc, bins=bins_prob)

ratio = np.divide(counts_signal, counts_background, out=np.zeros_like(counts_signal, dtype=float), where=counts_background!=0)

#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

#ax1.hist(prob_signal_nc, bins=bins_prob, histtype='step', lw=2.5, label='EM')
#ax1.hist(prob_background_nc, bins=bins_prob, histtype='step', lw=2.5, label='Hadronic')
#ax1.hist(prob_background2_nc, bins=bins_prob, histtype='step', label=r'$\nu_\mu$')
#ax1.set_ylabel('Count')
#ax1.legend()
#ax1.grid(True)

#ax2.plot(bins_prob[:-1] + 0.02*0.5, ratio, marker='o', linestyle='-', color='black', lw=2.5)
#ax2.axhline(y=1, c='r', ls='--')
#ax2.set_xlabel('Probability Score')
#ax2.set_ylabel('Signal / Background')
#ax2.grid(True) 

plt.figure(figsize=(8, 6))

plt.hist(prob_signal_nc, bins=bins_prob, histtype='step', lw=2.5, label='EM')
plt.hist(prob_background_nc, bins=bins_prob, histtype='step', lw=2.5, label='Hadronic')
plt.hist(prob_background2_nc, bins=bins_prob, histtype='step', lw=2.5, label=r'$\nu_\mu$')
#plt.set_ylabel('Count')
plt.legend()

#ax2.plot(bins_prob[:-1] + 0.02*0.5, ratio, marker='o', linestyle='-', color='black', lw=2.5)
#ax2.axhline(y=1, c='r', ls='--')
plt.xlabel('Probability Score')
plt.ylabel('Count')
#ax2.grid(True) 

plt.tight_layout()
plt.savefig('probability_score_model_nc_classifier_with_numu.pdf', bbox_inches='tight', dpi=200)
plt.clf()

#model 8
counts_signal, bin_edges = np.histogram(prob_signal_numu, bins=bins_prob)
counts_background, _ = np.histogram(prob_background_numu, bins=bins_prob)

ratio = np.divide(counts_signal, counts_background, out=np.zeros_like(counts_signal, dtype=float), where=counts_background!=0)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

ax1.hist(prob_signal_numu, bins=bins_prob, histtype='step', lw=2.5, label='EM')
ax1.hist(prob_background_numu, bins=bins_prob, histtype='step', lw=2.5, label=r'$\nu_\mu$')
#ax1.hist(prob_background2_numu, bins=bins_prob, histtype='step', label='NC')
ax1.set_ylabel('Count')
ax1.legend()
#ax1.grid(True)

ax2.plot(bins_prob[:-1] + 0.02*0.5, ratio, marker='o', linestyle='-', color='black', lw=2.5)
ax2.axhline(y=1, c='r', ls='--')
ax2.set_xlabel('Probability Score')
ax2.set_ylabel('Signal / Background')
#ax2.grid(True) 

plt.tight_layout()
plt.savefig('probability_score_model_numu_classifier.pdf', bbox_inches='tight', dpi=200)
plt.clf()