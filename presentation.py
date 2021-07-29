import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def draw_labels(X, y):
    # figure = plt.figure(figsize=(20, 6))
    # just plot the dataset first
    cm_colors = plt.set_cmap('Dark2')
    #
    # Plot the training points
    #  ax.set_title('train data')
    le = preprocessing.LabelEncoder()
    le.fit(y)
    plt.scatter(X['LON'], X['LAT'], c=le.transform(y), cmap=cm_colors, edgecolors='k')
    plt.show()

'''    

    # Plot the testing points
    le.fit(y_test)
    # ax.scatter(X_test['LAT'], X_test['LON'], c=le.transform(y_test), cmap=cm_blues, alpha=0.6, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ax.set_title(f'{field_name} - {clf_name}')
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')

    # ax = plt.subplot(1, 1 ,1)
    # clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    cm = plt.cm.RdBu
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    # ax.scatter(X_train[:, 0], X_train[:, 1], c=le.transform(y_train), cmap=cm_blues, edgecolors='k')
    # # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=le.transform(y_test), cmap=cm_blues , edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    plt.tight_layout()
    plt.show()

def draw_dataset(X_train, X_test, y_train, y_test, field_name, clf, clf_name, score):
    figure = plt.figure(figsize=(27, 9))

    h = .02  # step size in the mesh
    x_min, x_max = X_train['LAT'].min() - .5, X_train['LAT'].max() + .5
    y_min, y_max = X_train['LON'].min() - .5, X_train['LON'].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    # cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    cm_blues = plt.set_cmap('Blues')
    ax = plt.subplot(1,1,1)#  plt.subplot(len(field_names), len(clfs) + 1, i)
    # Plot the training points
    ax.set_title('train data')
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    ax.scatter(X_train['LAT'], X_train['LON'], c=le.transform(y_train), cmap=cm_blues, edgecolors='k')
    # Plot the testing points
    le.fit(y_test)
    ax.scatter(X_test['LAT'], X_test['LON'], c=le.transform(y_test), cmap=cm_blues, alpha=0.6, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ax.set_title(f'{field_name} - {clf_name}')
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')

    # ax = plt.subplot(1, 1 ,1)
    # clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    cm = plt.cm.RdBu
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    # ax.scatter(X_train[:, 0], X_train[:, 1], c=le.transform(y_train), cmap=cm_blues, edgecolors='k')
    # # Plot the testing points
    # ax.scatter(X_test[:, 0], X_test[:, 1], c=le.transform(y_test), cmap=cm_blues , edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    plt.tight_layout()
    plt.show()
'''