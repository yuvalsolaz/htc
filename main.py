import sys
import pandas as pd
from dataset import load_data, create_dataset, add_random_field
from classifier import classify, classifiers, names

from presentation import draw

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
        #print(X_train.head(), y_train.head())

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            print(f'start classify {field_name} label with {name}...')
            score = classify(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, clf=clf, clf_name=name)
            print ( f'{name} finished with {score} score')
            results.append({'field':field_name,
                           'classifier':name,
                           'score':score})

    print(pd.DataFrame(results))


