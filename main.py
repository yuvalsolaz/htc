import sys
import pandas as pd
import matplotlib.pyplot as plt

from dataset import load_data, create_dataset, add_random_field, geo
from classifier import classify, classifiers
from presentation import draw_labels, draw_decision_boundary

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'usage python {sys.argv[0]} <table_name> <field_name>')
        exit(1)

    table_name = sys.argv[1]
    table = load_data(table_name)
    field_names = sys.argv[2:]

    results = []
    fig, axs = plt.subplots(len(field_names), 1, 'all')
    for i, field_name in enumerate(field_names):
        print(f'creating data set from {field_name} field on {table_name} table...')
        X_train, X_test, y_train, y_test = create_dataset(table=table,field_name=field_name)
        # draw_labels(X=X_train, y=y_train, ax=axs[0], title='train')
        draw_labels(X=X_test , y=y_test , ax=axs[i], title='test')

        # select classifier:
        clf = classifiers[0]
        clf_name = str(clf)
        print(f'start classify {field_name} label with {clf_name}...')
        score = classify(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, clf=clf)
        print ( f'{clf_name} finished with {score} score')
        results.append({'field_name':field_name, 'classifier':clf, 'calssifier_name':clf_name, 'score':score})
        print(f'set {field_name} field as label')
        draw_decision_boundary(X=table[geo], clf=clf, ax=axs[i],
                               title = f'decision boundaries for {clf_name} with {field_name} label')
    print(pd.DataFrame(results))
    plt.show()
    pass


