import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_hdf('/data/user/akatil/electron_neutrino/for_real/dataset_complete/BDT/BDT_all.h5', key='data')

data = df #.sample(420000, random_state=42)

y = data.Label

features = ['CogDistance', 'Beta1Reco', 'Beta2Reco', 'Beta3Reco', 'Beta4Reco', 'Beta5Reco', 'tres_median', 'tres_iqr', 'dist_median', 'dist_iqr', 'ang_median', 'ang_iqr']

X = data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

depths = range(1, 101)
scores = []
importances = []

for depth in depths:
    print(f'depth is {depth}')
    model = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=1)
    score = np.median(cross_val_score(model, X, y, cv=5))

    scores.append(score)

    model.fit(train_X, train_y)
    importance = model.feature_importances_
    importances.append(importance)

importance_df = pd.DataFrame(importances, columns=features, index=depths)
importance_df.index.name = 'Max Depth' 

plt.plot(depths, scores, 'o--', lw=2)
plt.xlabel('Maximum Depth')
plt.ylabel('Cross Validation Accuracy')
plt.tight_layout()
plt.savefig("DecisionTreeClassifier_Entropy.png", dpi=300)
plt.clf() 


for feature in features:
    plt.plot(importance_df.index, importance_df[feature], 'o--', lw=2, label = feature) 

plt.xlabel("Maximum Depth")
plt.ylabel("Feature Importance")
plt.legend()
plt.tight_layout()
plt.savefig("DecisionTreeClassifier_Entropy_Feature_Importance.png", dpi=300)
plt.clf()

print(importance_df.round(4))

#print(score, len(score))

#model.fit(train_X, train_y)

#val_predictions = model.predict(val_X)

#diff = val_predictions - val_y

#percent_accurate = 100 * (len(diff) - len(diff[diff != 0]))/len(diff)

#print(f'% of events that are accurate: {percent_accurate}')

#print("Accuracy:", metrics.accuracy_score(val_y, val_predictions))
