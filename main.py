import sys
import pandas as pd
from dataset import load_data, create_dataset, split_dataset, GEO_FIELDS
from classifier import classify, classifiers

DRAW_FIGURE = False

if DRAW_FIGURE:
    import matplotlib.pyplot as plt
    from presentation import draw_labels, draw_decision_boundary


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'usage python {sys.argv[0]} <table_name> <fields names> optional')
        exit(1)

    table_name = sys.argv[1]
    table = load_data(table_name)

    if len(sys.argv) > 2:
        field_names = sys.argv[2:]
    else:
        field_names = set(table.columns[1:]) - set(GEO_FIELDS)

    results = []
    if DRAW_FIGURE:
        fig, axs = plt.subplots(len(field_names), 1, 'all')

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
        if DRAW_FIGURE:
            draw_labels(X=X_train, y=y_train, ax=axs[0], title='train')
            draw_labels(X=X_test , y=y_test , ax=axs[i], title='test')

        # select classifier:
        clf = classifiers[0]
        clf_name = str(clf)
        print(f'start classify {field_name} label with {clf_name}...')
        score = classify(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, clf=clf)
        print ( f'{clf_name} finished with {score} score')
        results.append({'field_name':field_name,
                        'classes_count':classes_count,
                        'classifier_name':clf_name,
                        'score':score})
        if DRAW_FIGURE:
            draw_decision_boundary(X=X, clf=clf, ax=axs[i],
                                   title = f'decision boundaries for {clf_name} with {field_name} label {score} score')
    print(pd.DataFrame(results))
    if DRAW_FIGURE:
        plt.show()
    pass


