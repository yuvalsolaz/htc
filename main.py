import sys
import pandas as pd
import matplotlib.pyplot as plt

from dataset import load_data, create_dataset, add_random_field
from classifier import classify, classifiers, names

from presentation import draw_labels

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'usage python {sys.argv[0]} <table_name> <field_name>')
        exit(1)

    table_name = sys.argv[1]
    table = load_data(table_name)
    field_names = sys.argv[2:]

    results = []
    for field_name in field_names:
        if field_name not in table.columns:
            table = add_random_field(df=table,field_name=field_name,categories=['XXX', 'YYY', 'ZZZ', 'XYZ', 'UNKNOWN'])

        print(f'creating data set from {field_name} field on {table_name} table...')
        X_train, X_test, y_train, y_test = create_dataset(table=table,field_name=field_name)

        figure = plt.figure(figsize=(12,12))
        fig, axs = plt.subplots(1, 2, 'all')
        draw_labels(X=X_train, y=y_train, ax=axs[0], title='train')
        draw_labels(X=X_test , y=y_test , ax=axs[1], title='test')
        plt.show()

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            print(f'start classify {field_name} label with {name}...')
            score = classify(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, clf=clf, clf_name=name)
            print ( f'{name} finished with {score} score')
            results.append({'field_name':field_name,
                            'classifier':clf,
                            'calssifier_name':name,
                            'score':score})

    print(pd.DataFrame(results))
    result = results[0]
    # draw(X_train, X_test, y_train, y_test,
    #      field_name=result['field_name'],
    #      clf = result['classifier'],
    #      clf_name=result['calssifier_name'],
    #      score=results[0]['score'])


