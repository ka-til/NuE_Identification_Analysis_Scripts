import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

df = pd.read_hdf('/data/user/akatil/electron_neutrino/for_real/dataset_complete/BDT/BDT_all.h5', key='data')

data = df #.sample(420000, random_state=42)

y = data.Label

features = ['CogDistance', 'Beta1Reco', 'Beta2Reco', 'Beta3Reco', 'Beta4Reco', 'Beta5Reco', 'tres_median', 'tres_iqr', 'dist_median', 'dist_iqr', 'ang_median', 'ang_iqr']

X = data[features]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state=0)

model = HistGradientBoostingClassifier(max_iter=100000, early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, random_state=0)
model.fit(train_X, train_y)


test_scores = [metrics.accuracy_score(test_y, y_pred) for y_pred in model.staged_predict(test_X)]

plt.plot(range(0, len(test_scores)), test_scores, 'o--', lw=2)
plt.xlabel('Boosting Stage')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('Gradient_Boosting_Classifier_Accuracy_plot.png', dpi=300)
plt.clf()

#loss curve
plt.plot(model.train_score_, 'o--', lw=2, label='Train')
plt.plot(model.validation_score_, 'o--', lw=2, label='Validation')
plt.xlabel('Boosing Stage')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('Gradient_Boosting_Classifier_Loss_plot.png', dpi=300)
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
plt.savefig('Gradient_Boosting_Classifier_ROC_curve.png', dpi=300)
plt.clf()

#importance
#importances = model.feature_importances_

#importance_df = pd.DataFrame(importances, columns=features)

#print(importance_df.round(3))
