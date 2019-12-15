# import moduels
import pandas as pd
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# end import

#
# import data
filename = './section_3/iris.data.csv'
names = ['separ-length', 'separ-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(filename, names=names)

#
# Observe given data
# print(dataset.shape)  # Data dimension
# print(dataset.head(10))  # Observe top 10 rows
# print(dataset.describe())  # Observe description
# print(dataset.groupby('class').size())  # Observer by group

#
# Data visualization

# dataset.plot(kind='box', subplots=True, layout=(
#     2, 2), sharex=False, sharey=False)  # Single variable boxplot

# dataset.hist() # Single variable histogram

# pd.plotting.scatter_matrix(dataset) # Multi variable matrix
# pyplot.show()

#
# Seperate data set
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.2
seed = 7

X_train, X_validation, Y_train, Y_validatation = train_test_split(
    X, Y, test_size=validation_size, random_state=seed
)

# #
# # Algorithm estimation
# models = {}
# models['LR'] = LogisticRegression()
# models['LDA'] = LinearDiscriminantAnalysis()
# models['KNN'] = KNeighborsClassifier()
# models['CART'] = DecisionTreeClassifier()
# models['NB'] = GaussianNB()
# models['SVM'] = SVC()

# results = []
# for key in models:
#     kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#     cv_results = cross_val_score(
#         models[key], X_train, Y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_results)
#     print(f"{key}: {cv_results.mean()} {cv_results.std()}")

#
# Generate Model
svm = SVC()
svm.fit(X=X_train, y=Y_train)
PREDICTIONS = svm.predict(X_validation)

print(accuracy_score(Y_validatation, PREDICTIONS))
print(confusion_matrix(Y_validatation, PREDICTIONS))
print(classification_report(Y_validatation, PREDICTIONS))
