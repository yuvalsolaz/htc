import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors",
         # "Linear SVM",
         # "RBF SVM",
         # "Gaussian Process",
         # "Decision Tree",
         # "Random Forest",
         # "Neural Net",
         # "AdaBoost",
         # "Naive Bayes",
         # "QDA"
         ]

classifiers = [KNeighborsClassifier(3),
               # SVC(kernel="linear", C=0.025),
               # SVC(gamma=2, C=1),
               # GaussianProcessClassifier(1.0 * RBF(1.0)),
               # DecisionTreeClassifier(max_depth=5),
               # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               # MLPClassifier(alpha=1, max_iter=1000),
               # AdaBoostClassifier(),
               # GaussianNB(),
               # QuadraticDiscriminantAnalysis()
               ]


def classify(X_train, X_test, y_train, y_test):

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f'{name} - {score}')

