import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

df = pd.read_hdf('../dataset_complete/BDT/BDT_all.h5', key='data')

data = df #.sample(420000, random_state=42)

y = data.Label

features = ['CogDistance', 'Beta1Reco', 'Beta2Reco', 'Beta3Reco', 'Beta4Reco', 'Beta5Reco', 'tres_median', 'dist_median', 'ang_median']

X = data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=1)
score = cross_val_score(model, X, y, cv=5)

print(score, len(score))

model.fit(train_X, train_y)

val_predictions = model.predict(val_X)

diff = val_predictions - val_y

percent_accurate = 100 * (len(diff) - len(diff[diff != 0]))/len(diff) 

print(f'% of events that are accurate: {percent_accurate}')

print("Accuracy:", metrics.accuracy_score(val_y, val_predictions))

#plt.figure(figsize=(20, 10)) 
#plot_tree(model)  
#plt.savefig("tree_visualization_classifier.png", dpi=300)
#plt.close() 
