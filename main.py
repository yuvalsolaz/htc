import sys
from dataset import load_data, create_dataset
from classifier import classify


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'usage python {sys.argv[0]} <table_name> <field_name>')
        exit(1)

    table_name = sys.argv[1]
    field_name = sys.argv[2]
    print(f'creating data set from {field_name} field on {table_name} table...')
    table = load_data(table_name)
    X_train, X_test, y_train, y_test = create_dataset(table=table,field_name=field_name)
    print(X_train.head(), y_train.head())

    print('start classify...')
    classify(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

