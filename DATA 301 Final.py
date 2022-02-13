import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as preprocessing


df = pd.read_csv("/content/drive/MyDrive/DATA301/data/DATA301.csv")
df = df.rename({'Diabetes_binary':'Target'}, axis=1)


df_bin = df[['Target', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']]
for c in range(len(df_bin.columns)):
  for r in range(len(df_bin)):
    #Indexing Method from pandas User Guide
    if df_bin.iloc[(r, c)] == 0:
      df_bin.iloc[(r, c)] = -1
df_bin


df_other = df[['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']]
df_other = preprocessing.StandardScaler().fit_transform(df_other)
df_other = pd.DataFrame(df_other, columns = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income'])
df_other

df = pd.concat([df_bin, df_other], axis = 1)
df

s = df.corr()["Target"].sort_values(ascending = True)
s.plot.barh(color = 'b')
plt.title("Correlations with Response Variable")
plt.xlabel("Correlation (r)")
plt.ylabel('Explanatory Variables')

########### K-Nearest Neighbors ##############

#Use explanatories with correlation with response > .2 
x = df[["HeartDiseaseorAttack", "PhysHlth", "DiffWalk", "Age", "HighChol", "BMI", "HighBP", "GenHlth"]] 
y = df['Target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
model = KNeighborsClassifier(n_neighbors = 15)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


knn_acc = metrics.accuracy_score(y_test, y_pred)
knn_f1 = metrics.f1_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

neighbors = range(1,30)
train_results = []
test_results = []
for n in neighbors:
   model = KNeighborsClassifier(n_neighbors=n)
   model.fit(x_train, y_train)
   train_pred = model.predict(x_train)
   thresholds = metrics.accuracy_score(y_train, train_pred)
   train_results.append(thresholds)
   y_pred = model.predict(x_test)
   threshold = metrics.accuracy_score(y_test, y_pred)
   test_results.append(threshold)
   
line1 = plt.plot(neighbors, train_results, 'b', label='Train Accuracy')
line2, = plt.plot(neighbors, test_results, 'r', label='Test Accuracy')
plt.ylabel('Accuracy score')
plt.xlabel('n_neighbors')
plt.title("KNN Accuracy Score")
plt.legend()
plt.show()


from sklearn.pipeline import Pipeline

# evaluate pca with KNearestNeighbors algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# define dataset
# define the pipeline
steps = [('pca', PCA(n_components=4)), ('m', KNeighborsClassifier(n_neighbors= 15))]
model = Pipeline(steps=steps)
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
acc_pca = mean(n_scores) 
print(classification_report(y_test, y_pred))
model

neighbors = range(1,8)

train_results = []
test_results = []
for n in neighbors:

  steps = [('pca', PCA(n_components=n)), ('m', KNeighborsClassifier(n_neighbors = 15))]
  model = Pipeline(steps=steps)
  model.fit(x_train, y_train)
  train_pred = model.predict(x_train)
  thresholds = metrics.accuracy_score(y_train, train_pred)
  train_results.append(thresholds)
  y_pred = model.predict(x_test)
  threshold = metrics.accuracy_score(y_test, y_pred)
  test_results.append(threshold)

line1 = plt.plot(neighbors, train_results, 'b', label='Train Accuracy')
line2 = plt.plot(neighbors, test_results, 'r', label='Test Accuracy')
plt.ylabel('Accuracy score')
plt.xlabel('PCA Components')
plt.title("KNN w/ PCA Accuracy Scores")
plt.legend()
plt.show()



##### Naive Bayes ##############

from sklearn.preprocessing import StandardScaler
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
ac_nb = metrics.accuracy_score(y_test,y_pred)
f1_nb = metrics.f1_score(y_test,y_pred)
print(classification_report(y_test, y_pred))
ac_nb

#Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred)
sn.heatmap(cm)
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
#(0,0) => TN 
#(0,1) => FP
#(1,0) => FN 
#(1,1) => TP

#ML model comparison 
final = [knn_acc, acc_pca, ac_nb]
bars = ("KNN", "KNN w/PCA", "Naive Bayes")
y_pos = np.arange(len(bars))
plt.ylim(0, .8) 
plt.bar(y_pos, final,color=(0.9, 0.9, 0.9),  edgecolor='blue')
plt.xticks(y_pos, bars)

plt.ylabel("Accuracy Score")
plt.title("Comparison of Differing Algorithms and Their Implementations")
plt.show()

nums = [0.604,0.71,0.728,0.73,0.749,0.758]
labs = ["K-Means Clustering", "Naive Bayes", "KNN w/ PCA", "KNN", "Neural Network", "Random Forests"] 
y_len = np.arange(len(nums)) 

plt.barh(y_len, nums, color = (0.9,0.9,0.9), edgecolor = "blue")
plt.yticks(y_len, labs)
plt.title("Comparing Accuracies of Algorithms")
plt.xlabel("Accuracy Scores") 
