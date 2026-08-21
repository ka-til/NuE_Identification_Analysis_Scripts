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

df = pd.read_hdf('/data/user/akatil/electron_neutrino/for_real/dataset_complete/BDT/BDT_all.h5', key='data')

data = df #.sample(420000, random_state=42)

y = data.Label

features = ['RecoEnergy', 'TotalCharge', 'CogDistance', 'Beta1Reco', 'Beta2Reco', 'Beta3Reco', 'Beta4Reco', 'Beta5Reco', 'tres_median', 'tres_iqr', 'dist_median', 'dist_iqr', 'ang_median', 'ang_iqr']

X = data[features]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state=0)

model = HistGradientBoostingClassifier(learning_rate=0.01, max_iter=10000, early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, random_state=0)
model.fit(train_X, train_y)

joblib.dump(model, 'first_iter.joblib')

test_scores = [metrics.accuracy_score(test_y, y_pred) for y_pred in model.staged_predict(test_X)]

#calculate feature importance
#importances = model.feature_importances_

#Calculate permutation importance
result = permutation_importance(model, test_X, test_y, n_repeats=30, random_state=0)
perm_importances = result.importances_mean
perm_std = result.importances_std

#Feature importance
#plt.barh(features, importances)
#plt.xlabel("Feature Importance")
#plt.tight_layout()
#plt.savefig('feature_importance_plot_first_iter.png', dpi=300)
#plt.clf()

#Permutation importance
plt.barh(features, perm_importances, xerr=perm_std)
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.savefig('permutation_importance_plot_first_iter.png', dpi=300)
plt.clf()

#Accuracy curve
plt.plot(range(0, len(test_scores)), test_scores, 'o--', lw=2)
plt.xlabel('Boosting Stage')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('accuracy_plot_first_iter.png', dpi=300)
plt.clf()

#loss curve
plt.plot(model.train_score_, 'o--', lw=2, label='Train')
plt.plot(model.validation_score_, 'o--', lw=2, label='Validation')
plt.xlabel('Boosing Stage')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('loss_plot_first_iter.png', dpi=300)
plt.clf()

#ROC-AUC curve
prob_y = model.predict_proba(test_X)[:, 1]
false_positive, true_positive, _ = metrics.roc_curve(test_y, prob_y)
roc_auc = metrics.auc(false_positive, true_positive)

plt.plot(false_positive, true_positive, 'o--', lw=2, label = f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('roc_plot_first_iter.png', dpi=300)
plt.clf()

#PR Curve
prob_y = model.predict_proba(test_X)[:, 1]
precision, recall, _ = metrics.precision_recall_curve(test_y, prob_y)
pr_auc = metrics.auc(recall, precision)
baseline = sum(test_y) / len(test_y)

plt.plot(recall, precision, 'o--', lw=2, label = f'AUC = {pr_auc:.2f}')
plt.axhline(y=baseline, c='k', ls='--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.savefig('pr_plot_first_iter.png', dpi=300)
plt.clf()

#importance
#importances = model.feature_importances_

#importance_df = pd.DataFrame(importances, columns=features)

#print(importance_df.round(3))
