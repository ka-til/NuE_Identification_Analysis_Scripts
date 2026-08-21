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

df = pd.read_hdf('/data/user/akatil/electron_neutrino/for_real/dataset_complete/BDT/BDT_all_updated.h5', key='data')

df.dropna(inplace=True)

data_7 = df[(df['TrueEnergy']>=2)&(df['TrueEnergy']<=15)&(df['TrueZenith']<=-0.3)]
data_7 = data_7.groupby('Label').sample(90000, random_state=0)

y_7 = data_7.Label

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

X_7 = data_7[features]

train_X_7, test_X_7, train_y_7, test_y_7 = train_test_split(X_7, y_7, test_size = 0.2, random_state=0)

model_7 = joblib.load('/data/user/akatil/electron_neutrino/for_real/analysis_chain/bdt_scripts/submit_scripts/seventh_iter.joblib')

prob_y_7 = model_7.predict_proba(test_X_7)[:, 1]
false_positive_7, true_positive_7, _ = metrics.roc_curve(test_y_7, prob_y_7)
roc_auc_7 = metrics.auc(false_positive_7, true_positive_7)

reco_energy_bins = np.linspace(min(test_X_7['RecoEnergy']), max(test_X_7['RecoEnergy']), 50)
plt.hist(test_X_7['RecoEnergy'][test_y_7 == 1], bins=reco_energy_bins, histtype='step', label='CC')
plt.hist(test_X_7['RecoEnergy'][test_y_7 == 0], bins=reco_energy_bins, histtype='step', label='NC')
plt.hist(test_X_7['RecoEnergy'][prob_y_7 <= 0.2], bins=reco_energy_bins, histtype='step', label='Probability Score Cut')
plt.legend()
plt.xlabel('Reco Energy [GeV]')
plt.tight_layout()
plt.savefig('reco_energy_prob_cut_seventh_iter', dpi=300)
plt.clf()

total_charge_bins = np.linspace(min(test_X_7['TotalCharge']), max(test_X_7['TotalCharge']), 50)
plt.hist(test_X_7['TotalCharge'][test_y_7 == 1], bins=total_charge_bins, histtype='step', label='CC')
plt.hist(test_X_7['TotalCharge'][test_y_7 == 0], bins=total_charge_bins, histtype='step', label='NC')
plt.hist(test_X_7['TotalCharge'][prob_y_7 <= 0.2], bins=total_charge_bins, histtype='step', label='Probability Score Cut')
plt.legend()
plt.xlabel('Total Charge')
plt.tight_layout()
plt.savefig('total_charge_prob_cut_seventh_iter', dpi=300)
plt.clf()

beta1_bins = np.linspace(min(test_X_7['Beta1Reco']), max(test_X_7['Beta1Reco']), 50)
plt.hist(test_X_7['Beta1Reco'][test_y_7 == 1], bins=beta1_bins, histtype='step', label='CC')
plt.hist(test_X_7['Beta1Reco'][test_y_7 == 0], bins=beta1_bins, histtype='step', label='NC')
plt.hist(test_X_7['Beta1Reco'][prob_y_7 <= 0.2], bins=beta1_bins, histtype='step', label='Probability Score Cut')
plt.legend()
plt.xlabel('Beta1 Reco')
plt.tight_layout()
plt.savefig('beta1_prob_cut_seventh_iter', dpi=300)
plt.clf()


