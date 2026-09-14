import os
import pandas as pd
from sqlalchemy import create_engine

def setup_database():
    base_dir = os.getcwd() 
    
    db_path = '/app/src/database/real_estate.db'
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    engine = create_engine(f'sqlite:///{db_path}')

    files_to_tables = {
        'city_indexes.csv': 'city_indexes',
        'city_search_index.csv': 'city_search_index',
        'land_transactions.csv': 'land_transactions',
        'land_transactions_nearby_sectors.csv': 'land_transactions_nearby_sectors',
        'new_house_transactions.csv': 'new_house_transactions',
        'new_house_transactions_nearby_sectors.csv': 'new_house_transactions_nearby_sectors',
        'pre_owned_house_transactions.csv': 'pre_owned_house_transactions',
        'pre_owned_house_transactions_nearby_sectors.csv': 'pre_owned_house_transactions_nearby_sectors',
        'sector_POI.csv': 'sector_poi',
    }

    for filename, table_name in files_to_tables.items():
        file_path = os.path.join(base_dir, filename)
        
        if not os.path.exists(file_path):
            fallback_path = os.path.join(base_dir, 'data', 'sample', filename)
            if os.path.exists(fallback_path):
                file_path = fallback_path

        if not os.path.exists(file_path):
            print(f"Warning: {filename} not found (looked in {file_path}), skipping.")
            continue

        try:
            df = pd.read_csv(file_path)
            df.to_sql(table_name, con=engine, if_exists='replace', index=False)
            print(f"Loaded {filename} -> table '{table_name}' ({len(df):,} rows).")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    print(f"Database setup complete. Data has been stored in {db_path}.")

if __name__ == "__main__":
    setup_database()