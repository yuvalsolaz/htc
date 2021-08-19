import sys
import pandas as pd
import matplotlib.pyplot as plt

from presentation import draw_labels, draw_decision_boundary
from dataset import load_data, create_dataset, split_dataset, GEO_FIELDS
from classifier import classify, classifiers

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage python {sys.argv[0]} <table_name> <fields names> optional')
        exit(1)

    table_name = sys.argv[1]
    table = load_data(table_name)

    if len(sys.argv) > 2:
        field_names = sys.argv[2:]
    else:
        field_names = list(set(table.columns) - set(GEO_FIELDS) - set(['HASH','ID']))

    results = []
    fig, axs = plt.subplots(len(field_names), 2, 'none', figsize=(17,12))
    for i, field_name in enumerate(field_names):
        print(f'creating data set from {field_name} field on {table_name} table...')
        X,y = create_dataset(table=table, field_name=field_name)
        if X is None or y is None:
            continue

        classes_count = len(set(y))
        print(f'{field_name} have {classes_count} distinct values')
        if classes_count < 2:
            continue

        X_train, X_test, y_train, y_test = split_dataset(X, y)
        clf = classifiers[0]
        clf_name = str(clf)
        print(f'start classify {field_name} label with {clf_name}...')
        score = classify(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, clf=clf)
        print ( f'{clf_name} finished with {score} score')
        results.append({'field_name':field_name,
                        'classes_count':classes_count,
                        'classifier_name':clf_name,
                        'score':score})

        draw_labels(X=X_test , y=y_test , ax=axs[i,0], title='test')
        draw_decision_boundary(X=X, clf=clf, ax=axs[i,1],
                              title = f'decision line {field_name} label {classes_count} classes {score} score')
    print(pd.DataFrame(results))
    plt.show()
    pass


