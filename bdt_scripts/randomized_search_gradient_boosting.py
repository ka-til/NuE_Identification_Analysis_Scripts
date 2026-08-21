    from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

#df = pd.read_hdf('/data/user/akatil/electron_neutrino/for_real/dataset_complete/BDT/BDT_all_updated.h5', key='data')

df = pd.read_hdf('/data/user/akatil/electron_neutrino/for_real/dataset_complete/BDT/BDT_all_numu_nue.h5', key='data')

df.dropna(inplace=True)
data = df[(df['TrueEnergy']>=2)&(df['TrueEnergy']<=15)&(df['TrueZenith']<=-0.3)] #[df['TrueEnergy'] <= 15] #[(df['TrueEnergy']>=2)&(df['TrueEnergy']<=15)&(df['TrueZenith']<=-0.3)] #[df['RecoEnergy'] < 20] #.sample(420000, random_state=42)
data = data.groupby('Label').sample(94592, random_state=0) #data.groupby('Label').sample(200000, random_state=0) #data.groupby('Label').sample(200000, random_state=0)

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

param_dist = {'n_estimators': [20, 50, 80, 300, 500, 700, 1000],
            'learning_rate': [0.01, 0.05, 0.1, 0.275],
            'max_depth': [3, 4, 5, 6, 8],
            'min_samples_split': [2, 100, 500, 800, 1100],
            'min_samples_leaf': [1, 5, 8, 10, 15, 20],
            'max_features': [1, 5, 8, 11, 13]
            }

model = GradientBoostingClassifier(validation_fraction=0.2, n_iter_no_change=10, random_state=0)

random_search = RandomizedSearchCV(estimator=model,
            param_distributions = param_dist,
            n_iter=50,
            scoring='roc_auc',
            verbose=2,
            random_state=0)

random_search.fit(train_X, train_y)

best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

probs = best_model.predict_proba(test_X)[:, 1]
auc = roc_auc_score(test_y, probs)
print(f"Test ROC AUC: {auc:.4f}")
