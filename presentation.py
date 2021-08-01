import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

'''
    plot samples with colored labels 
'''
def draw_labels(X, y, ax, title):
    ax.set_title(title)
    cm_colors = plt.set_cmap('Dark2')
    le.fit(y)
    ax.scatter(X[:, 0], X[:, 1], c=le.transform(y), cmap=cm_colors, edgecolors='k')

def draw_decision_boundary(X, clf, ax, title):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    le.fit(Z)
    colors = le.transform(Z)
    Z = Z.reshape(xx.shape)
    ax.set_title(title)
    cm_colors = plt.set_cmap('Dark2')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.contourf(xx, yy, Z, c=colors, cmap= cm_colors, alpha=.8)