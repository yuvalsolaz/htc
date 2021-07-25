import sys
import pandas as pd
from sklearn.model_selection import train_test_split


geo = ['LAT', 'LON']

def load_data(data_file=r'data/ut-sample.csv'):
    print (f'loading {data_file}...')
    df = pd.read_csv(data_file)
    print(f'{df.shape} records loaded')
    return df


def create_dataset(table, field_name):
    table.dropna(subset=geo + [field_name], inplace=True)
    print(f'{table.shape} after drop nan')
    print ('set geo fields as X features')
    X = table[geo]
    print(f'set {field_name} field as label')
    y = table[field_name]
    test_size = .30
    print(f'split train test {(1-test_size)*100}% - {test_size* 100}%')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    return X_train, X_test, y_train, y_test







