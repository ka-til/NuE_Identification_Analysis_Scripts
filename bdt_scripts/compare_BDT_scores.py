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

data_5 = df[df['RecoEnergy'] < 20]
data_5 = data_5.groupby('Label').sample(200000, random_state=0)

data_6 = df
data_6 = data_6.groupby('Label').sample(200000, random_state=0)

data_7 = df[(df['TrueEnergy']>=2)&(df['TrueEnergy']<=15)&(df['TrueZenith']<=-0.3)]
data_7 = data_7.groupby('Label').sample(90000, random_state=0)

data_8 = df[df['TrueEnergy'] <= 15]
data_8 = data_8.groupby('Label').sample(200000, random_state=0)

y_5 = data_5.Label
y_6 = data_6.Label
y_7 = data_7.Label
y_8 = data_8.Label

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

X_5 = data_5[features]
X_6 = data_6[features]
X_7 = data_7[features]
X_8 = data_8[features]

train_X_5, test_X_5, train_y_5, test_y_5 = train_test_split(X_5, y_5, test_size = 0.2, random_state=0)
train_X_6, test_X_6, train_y_6, test_y_6 = train_test_split(X_6, y_6, test_size = 0.2, random_state=0)
train_X_7, test_X_7, train_y_7, test_y_7 = train_test_split(X_7, y_7, test_size = 0.2, random_state=0)
train_X_8, test_X_8, train_y_8, test_y_8 = train_test_split(X_8, y_8, test_size = 0.2, random_state=0)

model_5 = joblib.load('/data/user/akatil/electron_neutrino/for_real/analysis_chain/bdt_scripts/submit_scripts/fifth_iter.joblib')
model_6 = joblib.load('/data/user/akatil/electron_neutrino/for_real/analysis_chain/bdt_scripts/submit_scripts/sixth_iter.joblib')
model_7 = joblib.load('/data/user/akatil/electron_neutrino/for_real/analysis_chain/bdt_scripts/submit_scripts/seventh_iter.joblib')
model_8 = joblib.load('/data/user/akatil/electron_neutrino/for_real/analysis_chain/bdt_scripts/submit_scripts/eight_iter.joblib')

#ROC-AUC curve
prob_y_5 = model_5.predict_proba(test_X_7)[:, 1]
prob_y_6 = model_6.predict_proba(test_X_7)[:, 1]
prob_y_7 = model_7.predict_proba(test_X_7)[:, 1]
prob_y_8 = model_8.predict_proba(test_X_7)[:, 1]

prob_signal_5 = prob_y_5[test_y_7 == 1]
prob_background_5 = prob_y_5[test_y_7 == 0]

prob_signal_6 = prob_y_6[test_y_7 == 1]
prob_background_6 = prob_y_6[test_y_7 == 0]

prob_signal_7 = prob_y_7[test_y_7 == 1]
prob_background_7 = prob_y_7[test_y_7 == 0]

prob_signal_8 = prob_y_8[test_y_7 == 1]
prob_background_8 = prob_y_8[test_y_7 == 0]

bins_prob = np.arange(0, 1, 0.02)

#model 5
counts_signal, bin_edges = np.histogram(prob_signal_5, bins=bins_prob)
counts_background, _ = np.histogram(prob_background_5, bins=bins_prob)

ratio = np.divide(counts_signal, counts_background, out=np.zeros_like(counts_signal, dtype=float), where=counts_background!=0)

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

# Top plot: Histogram
ax1.hist(prob_signal_5, bins=bins_prob, histtype='step', label='CC')
ax1.hist(prob_background_5, bins=bins_prob, histtype='step', label='NC')
ax1.set_ylabel('Count')
ax1.legend()
ax1.grid(True)

# Bottom plot: S/B Ratio
ax2.plot(bins_prob[:-1] + 0.02*0.5, ratio, marker='o', linestyle='-', color='black')
ax2.axhline(y=1, c='r', ls='--')
ax2.set_xlabel('Probability Score')
ax2.set_ylabel('Signal / Background')
ax2.grid(True) 

plt.tight_layout()
plt.savefig('probability_score_ratio_combined_model_5.png', dpi=300)
plt.clf()

#model 6
counts_signal, bin_edges = np.histogram(prob_signal_6, bins=bins_prob)
counts_background, _ = np.histogram(prob_background_6, bins=bins_prob)

ratio = np.divide(counts_signal, counts_background, out=np.zeros_like(counts_signal, dtype=float), where=counts_background!=0)

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

# Top plot: Histogram
ax1.hist(prob_signal_6, bins=bins_prob, histtype='step', label='CC')
ax1.hist(prob_background_6, bins=bins_prob, histtype='step', label='NC')
ax1.set_ylabel('Count')
ax1.legend()
ax1.grid(True)

# Bottom plot: S/B Ratio
ax2.plot(bins_prob[:-1] + 0.02*0.5, ratio, marker='o', linestyle='-', color='black')
ax2.axhline(y=1, c='r', ls='--')
ax2.set_xlabel('Probability Score')
ax2.set_ylabel('Signal / Background')
ax2.grid(True) 

plt.tight_layout()
plt.savefig('probability_score_ratio_combined_model_6.png', dpi=300)
plt.clf()

#model 7
counts_signal, bin_edges = np.histogram(prob_signal_7, bins=bins_prob)
counts_background, _ = np.histogram(prob_background_7, bins=bins_prob)

ratio = np.divide(counts_signal, counts_background, out=np.zeros_like(counts_signal, dtype=float), where=counts_background!=0)

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

# Top plot: Histogram
ax1.hist(prob_signal_7, bins=bins_prob, histtype='step', label='CC')
ax1.hist(prob_background_7, bins=bins_prob, histtype='step', label='NC')
ax1.set_ylabel('Count')
ax1.legend()
ax1.grid(True)

# Bottom plot: S/B Ratio
ax2.plot(bins_prob[:-1] + 0.02*0.5, ratio, marker='o', linestyle='-', color='black')
ax2.axhline(y=1, c='r', ls='--')
ax2.set_xlabel('Probability Score')
ax2.set_ylabel('Signal / Background')
ax2.grid(True) 

plt.tight_layout()
plt.savefig('probability_score_ratio_combined_model_7.png', dpi=300)
plt.clf()

#model 8
counts_signal, bin_edges = np.histogram(prob_signal_8, bins=bins_prob)
counts_background, _ = np.histogram(prob_background_8, bins=bins_prob)

ratio = np.divide(counts_signal, counts_background, out=np.zeros_like(counts_signal, dtype=float), where=counts_background!=0)

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

# Top plot: Histogram
ax1.hist(prob_signal_8, bins=bins_prob, histtype='step', label='CC')
ax1.hist(prob_background_8, bins=bins_prob, histtype='step', label='NC')
ax1.set_ylabel('Count')
ax1.legend()
ax1.grid(True)

# Bottom plot: S/B Ratio
ax2.plot(bins_prob[:-1] + 0.02*0.5, ratio, marker='o', linestyle='-', color='black')
ax2.axhline(y=1, c='r', ls='--')
ax2.set_xlabel('Probability Score')
ax2.set_ylabel('Signal / Background')
ax2.grid(True) 

plt.tight_layout()
plt.savefig('probability_score_ratio_combined_model_8.png', dpi=300)
plt.clf()