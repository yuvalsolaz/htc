import sys
import pandas as pd


geo = ['LAT', 'LON']

def load_data(data_file=r'data/ut-sample.csv'):
    print (f'loading {data_file}...')
    df = pd.read_csv(data_file)
    print(f'{df.shape} records loaded')
    return df


def create_dataset(table, field_name):
    df = pd.DataFrame()
    df[geo] = table[geo]
    df['label'] = table[field_name]
    return df

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'usage python {sys.argv[0]} <table_name> <field_name>')
        exit(1)

    table_name = sys.argv[1]
    field_name = sys.argv[2]
    print(f'creating data set from {field_name} field on {table_name} table...')
    table = load_data(table_name)
    df = create_dataset(table=table,field_name=field_name)
    print (df.head())

