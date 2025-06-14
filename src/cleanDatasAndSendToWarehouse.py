from clients.minio import get_minio_client
from clients.postgres import get_warehouse_engine
import os
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO
import unicodedata
from utils import debugDF, loadFile

# Load environment variables from .env file
load_dotenv()

BUCKET_NAME = os.environ["MINIO_BUCKET"]

# Connect to MinIO
client = get_minio_client()
# Connect to Postgres
engine = get_warehouse_engine()

# Utility functions

def remove_accents_df(df):
    def strip_accents(s):
        if isinstance(s, str):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        return s
    return df.applymap(strip_accents)

# Data cleaning helpers

def clean_and_rename_columns(df):
    # Drop the Etat saisie column
    df = df.drop(columns=['Etat saisie'])

    # Rename main columns to English
    rename_map = {
        'Code du département': 'department_code',
        'Libellé du département': 'department_name',
        'Inscrits': 'registered',
        'Abstentions': 'abstentions',
        '% Abs/Ins': 'pct_abstention',
        'Votants': 'voters',
        '% Vot/Ins': 'pct_voters',
        'Blancs': 'blank_votes',
        '% Blancs/Ins': 'pct_blank_registered',
        '% Blancs/Vot': 'pct_blank_voters',
        'Nuls': 'null_votes',
        '% Nuls/Ins': 'pct_null_registered',
        '% Nuls/Vot': 'pct_null_voters',
        'Exprimés': 'valid_votes',
        '% Exp/Ins': 'pct_valid_registered',
        '% Exp/Vot': 'pct_valid_voters',
    }
    df = df.rename(columns=rename_map)

    candidate_fields = [
        'Sexe', 'Nom', 'Prénom', 'Voix', '% Voix/Ins', '% Voix/Exp'
    ]
    normalized_fields = [
        'candidate_gender', 'candidate_lastname', 'candidate_firstname',
        'candidate_votes', 'candidate_pct_votes_registered', 'candidate_pct_votes_valid'
    ]
    # Find the index of the first candidate field
    first_candidate_idx = df.columns.get_loc(candidate_fields[0])
    # Build new columns list
    new_columns = list(df.columns)
    candidate_num = 1
    for i in range(first_candidate_idx, len(df.columns), len(candidate_fields)):
        for j, field in enumerate(normalized_fields):
            col_idx = i + j
            if col_idx < len(new_columns):
                new_columns[col_idx] = f"{field}_{candidate_num}"
        candidate_num += 1
    df.columns = new_columns
    return df

def onlyKeepTotal(df):
    # Only keep rows where sexe is 'T'
    df_filtered = df[df['sexe'] == 'T'].copy()
    # Drop the 'sexe' column
    df_filtered = df_filtered.drop(columns=['sexe'])
    return df_filtered

def prepare_df22(df22, df0921):
    # Extract mapping from libgeo to codgeo from df0921
    mapping = df0921[['libgeo', 'codgeo']].drop_duplicates().set_index('libgeo')['codgeo']
    # Rename columns to match df0921
    df22 = df22.rename(columns={
        'Département': 'libgeo',
        'Taux de chômage': 'tx_chom1564'
    })
    # Add year column
    df22['an'] = 2022
    # Map codgeo using libgeo
    df22['codgeo'] = df22['libgeo'].map(mapping)
    # Reorder columns to match df0921
    df22 = df22[['codgeo', 'libgeo', 'an', 'tx_chom1564']]
    # Convert tx_chom1564 to float, coercing errors (e.g., 'nd' to NaN)
    df22['tx_chom1564'] = pd.to_numeric(df22['tx_chom1564'], errors='coerce')
    return df22

# ETL functions

def loadElectionsData():
    df = pd.read_excel(BytesIO(loadFile(client, BUCKET_NAME, "elections-2022-depts-t1.xlsx")))
    df = remove_accents_df(df)
    df = clean_and_rename_columns(df)
    debugDF(df)

    # Save DataFrame to Postgres
    print("Dumping DataFrame to Postgres...")
    df.to_sql(
        'elections_2022_departments',
        con=engine,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=1000
    )
    print("DataFrame successfully dumped to Postgres.")

def loadChomageData():
    df0921 = pd.read_excel(BytesIO(loadFile(client, BUCKET_NAME, "chomage_2009-2021.xlsx")), skiprows=4)
    df0921 = remove_accents_df(df0921)
    debugDF(df0921)

    df_filtered = onlyKeepTotal(df0921)
    debugDF(df_filtered)

    # For 2022: load rows 3 to 104 (Excel is 1-based, pandas is 0-based, so skiprows=2, nrows=102)
    df22 = pd.read_excel(
        BytesIO(loadFile(client, BUCKET_NAME, "chomage_2022.xlsx")),
        sheet_name="Figure 2b",
        skiprows=2,
        nrows=102
    )
    df22 = remove_accents_df(df22)
    debugDF(df22)

    # Prepare df22 to match df0921 format, using mapping from df0921
    df22_prepared = prepare_df22(df22, df_filtered)
    debugDF(df22_prepared)

    # Merge the two DataFrames
    merged = pd.concat([df_filtered, df22_prepared], ignore_index=True)

    merged = merged.rename(columns={
        'codgeo': 'department_code',
        'libgeo': 'department_name',
        'an': 'year',
        'tx_chom1564': 'unemployment_rate'
    })

    merged = merged.sort_values(by=['department_code', 'year']).reset_index(drop=True)
    debugDF(merged)

    # Save DataFrame to Postgres
    print("Dumping DataFrame to Postgres...")
    merged.to_sql(
        'unemployment_rates',
        con=engine,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=1000
    )
    print("DataFrame successfully dumped to Postgres.")

def loadCrimeDatas():
    df = pd.read_csv(BytesIO(loadFile(client, BUCKET_NAME, "crime-datas.csv")), delimiter=';')
    df = df.drop(columns=['insee_pop_millesime', 'insee_log_millesime'])
    debugDF(df)
    df.to_sql(
        'crime_data',
        con=engine,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=1000
    )

def loadImmigrationDatas():
    df = pd.read_excel(BytesIO(loadFile(client, BUCKET_NAME, "insee_rp_hist_xxxx.xlsx")), sheet_name="Data", skiprows=4)
    df = remove_accents_df(df)
    debugDF(df)
    df.to_sql(
        'immigration_data',
        con=engine,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=1000
    )

def loadIncomeDatas():
    df = pd.read_excel(BytesIO(loadFile(client, BUCKET_NAME, "TCRD_022.xlsx")), sheet_name="DEP", skiprows=4, skipfooter=2).drop(columns=['1er décile (D1)', '9e décile (D9)']).rename(columns={'Unnamed: 0': 'code_dep', 'Unnamed: 1': 'lib_dep', 'Part des ménages fiscaux imposés (en %)': 'prct_taxed', 'Médiane': 'median'})
    # Remove the line where code_dep is 'M'
    df = df[df['code_dep'] != 'M']
    df = remove_accents_df(df)
    debugDF(df)
    df.to_sql(
        'income_data',
        con=engine,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=1000
    )

def main():
    loadElectionsData()
    loadChomageData()
    loadCrimeDatas()
    loadImmigrationDatas()
    loadIncomeDatas()

if __name__ == "__main__":
    main()
