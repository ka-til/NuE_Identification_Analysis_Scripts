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
false_positive_5, true_positive_5, _ = metrics.roc_curve(test_y_7, prob_y_5)
roc_auc_5 = metrics.auc(false_positive_5, true_positive_5)

prob_y_6 = model_6.predict_proba(test_X_7)[:, 1]
false_positive_6, true_positive_6, _ = metrics.roc_curve(test_y_7, prob_y_6)
roc_auc_6 = metrics.auc(false_positive_6, true_positive_6)

prob_y_7 = model_7.predict_proba(test_X_7)[:, 1]
false_positive_7, true_positive_7, _ = metrics.roc_curve(test_y_7, prob_y_7)
roc_auc_7 = metrics.auc(false_positive_7, true_positive_7)

prob_y_8 = model_8.predict_proba(test_X_7)[:, 1]
false_positive_8, true_positive_8, _ = metrics.roc_curve(test_y_7, prob_y_8)
roc_auc_8 = metrics.auc(false_positive_8, true_positive_8)

plt.plot(false_positive_5, true_positive_5, '--', lw=2, label = f'reco energy cut, AUC = {roc_auc_5:.2f}')
plt.plot(false_positive_6, true_positive_6, '--', lw=2, label = f'no cut, AUC = {roc_auc_6:.2f}')
plt.plot(false_positive_8, true_positive_8, '--', lw=2, label = f'true energy cut, AUC = {roc_auc_8:.2f}')
plt.plot(false_positive_7, true_positive_7, '--', lw=2, label = f'true energy and zenith cut, AUC = {roc_auc_7:.2f}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('compare_roc_plot_test_same_region.png', dpi=300)
plt.clf()

#Accuracy plots
test_scores_5 = [metrics.accuracy_score(test_y_7, y_pred_5) for y_pred_5 in model_5.staged_predict(test_X_7)]
test_scores_6 = [metrics.accuracy_score(test_y_7, y_pred_6) for y_pred_6 in model_6.staged_predict(test_X_7)]
test_scores_7 = [metrics.accuracy_score(test_y_7, y_pred_7) for y_pred_7 in model_7.staged_predict(test_X_7)]
test_scores_8 = [metrics.accuracy_score(test_y_7, y_pred_8) for y_pred_8 in model_8.staged_predict(test_X_7)]

plt.plot(range(0, len(test_scores_5)), test_scores_5, '--', lw=2, label= f'reco energy cut')
plt.plot(range(0, len(test_scores_6)), test_scores_6, '--', lw=2, label= f'no cut')
plt.plot(range(0, len(test_scores_8)), test_scores_8, '--', lw=2, label= f'true energy cut')
plt.plot(range(0, len(test_scores_7)), test_scores_7, '--', lw=2, label= f'true energy and zenith cut')
plt.legend()
plt.xlabel('Boosting Stage')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('compare_accuracy_plot_test_same_region.png', dpi=300)
plt.clf()













