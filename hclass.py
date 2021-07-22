import pandas as pd
from enum import Enum
import numpy.random as random

'''

'''
def load_data(data_file=r'data/ut-sample.csv'):
    print (f'loading {data_file}...')
    df = pd.read_csv(data_file)
    print(f'{df.shape} records loaded')
    return df

def random_field(field_name, categories:list):
    def choose_category(categories):
        index = random.randint(low=0, high=len(categories))
        return categories[index]

    df[field_name] = df.apply(lambda x : choose_category(categories=categories),axis=1)
    return df

if __name__ == '__main__':
    df = load_data()
    print(df.head())
    df = random_field(field_name='rand',categories=['AAA','NNN','XXX','WWW'])
    print(df.head())
    df.to_csv(r'data/ut-rand.csv')

