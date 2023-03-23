import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load files into DataFrame

    Args:
        messages_filepath (str): messages table filepath
        categories_filepath (str): categories table filepath

    Returns:
        DataFrame: Concatenated dataframe of two tables
    """    
    return pd.read_csv(messages_filepath).join(pd.read_csv(categories_filepath).set_index('id'), on='id')


def clean_data(df):
    """Clean loaded data

    Args:
        df (DataFrame): Loaded data

    Returns:
        DataFrame: Cleaned data
    """    
    df.categories = df.categories.str.split(';').apply(lambda x: [category.split(
        '-')[0] for category in x if category.split('-')[-1] == '1'])
    df = df.join(df.categories.str.join('*').str.get_dummies(sep='*'))
    df = df.drop(columns='categories').drop_duplicates()
    return df


def save_data(df, database_filename):
    """Save data to a database

    Args:
        df (DataFrame): Data to save
        database_filename (str): db filepath
    """    
    engine = create_engine('sqlite:///{0}'.format(database_filename), echo=True)
    sqlite_connection = engine.connect()
    sqlite_table = "disaster_response"
    df.to_sql(sqlite_table, sqlite_connection, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
