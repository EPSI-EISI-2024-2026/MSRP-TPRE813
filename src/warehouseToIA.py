import os
import pandas as pd
from clients.postgres import get_warehouse_engine, get_iamart_engine
from clients.minio import get_minio_client
import re
import io

from utils import debugDF, loadFile

warehouse = get_warehouse_engine()
iamart = get_iamart_engine()
minio = get_minio_client()

DATAMART_BUCKET_NAME = 'datamart'
BUCKET_NAME = os.environ["MINIO_BUCKET"]

def main():
    electionDf = pd.read_sql_table('elections_2022_departments', warehouse)

    # Transform candidate columns as requested
    candidate_pattern = re.compile(r'candidate_gender_(\d+)')
    candidate_ids = [m.group(1) for col in electionDf.columns for m in [candidate_pattern.match(col)] if m]
    # Prepare a list of new columns and their values for all candidates
    new_cols = {}
    for cid in candidate_ids:
        gender_col = f'candidate_gender_{cid}'
        last_col = f'candidate_lastname_{cid}'
        first_col = f'candidate_firstname_{cid}'
        votes_col = f'candidate_votes_{cid}'
        # Build new column name using the values from the row, with votes_ prefix
        col_names = 'votes_' + electionDf[gender_col].astype(str) + '-' + electionDf[last_col].astype(str) + '-' + electionDf[first_col].astype(str)
        for idx, col_name in enumerate(col_names):
            new_cols.setdefault(col_name, [None]*len(electionDf))
            new_cols[col_name][idx] = electionDf[votes_col].iloc[idx]
    # Assign all new columns at once
    for col_name, values in new_cols.items():
        electionDf[col_name] = values
    # Drop all original candidate columns
    drop_cols = []
    for cid in candidate_ids:
        drop_cols.extend([
            f'candidate_gender_{cid}',
            f'candidate_lastname_{cid}',
            f'candidate_firstname_{cid}',
            f'candidate_votes_{cid}',
            f'candidate_pct_votes_registered_{cid}',
            f'candidate_pct_votes_valid_{cid}'
        ])
    electionDf = electionDf.drop(columns=[col for col in drop_cols if col in electionDf.columns])
    # Add rows of electionDf to IAdatas as dicts
    IAdf = electionDf.copy()
    del electionDf  # Free resources

    unemploymentDf = pd.read_sql('SELECT * FROM unemployment_rates WHERE year=2022', warehouse)

    # Merge unemployment data into IAdatas based on department_name
    IAdf = IAdf.merge(
        unemploymentDf[['department_name', 'unemployment_rate']],
        on='department_name',
        how='left',
        suffixes=('', '_unemployment')
    )
    del unemploymentDf  # Free resources

    incomeDf = pd.read_sql_table('income_data', warehouse)
    incomeDf = incomeDf.rename(columns={'lib_dep': 'department_name', 'median': 'median_income', 'Nombre': 'households_count', 'prct_taxed': 'pct_taxed_households'})

    IAdf = IAdf.merge(
        incomeDf[['department_name', 'median_income', 'households_count', 'pct_taxed_households']],
        on='department_name',
        how='left',
        suffixes=('', '_income')
    )
    del incomeDf  # Free resources

    immigrationDf = pd.read_sql("SELECT libgeo as department_name, part_immigres as pct_immigrants FROM immigration_data WHERE an=2021", warehouse)

    IAdf = IAdf.merge(
        immigrationDf,
        on='department_name',
        how='left',
        suffixes=('', '_immigration')
    )
    del immigrationDf  # Free resources

    crimeDf = pd.read_sql('SELECT "Code_departement" as departement_code, indicateur, unite_de_compte, nombre, taux_pour_mille, insee_pop as population FROM crime_data WHERE annee=2022', warehouse)

    # Normalize indicateur for crime column names
    import unicodedata
    def normalize_str(s):
        if pd.isna(s):
            return s
        s = ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')
        return s.strip().lower().replace(' ', '_')
    crimeDf['indicateur'] = crimeDf['indicateur'].apply(normalize_str)
    crimeDf['unite_de_compte'] = crimeDf['unite_de_compte'].apply(normalize_str)
    crimeDf['crime_col'] = crimeDf['indicateur'] + '--(' + crimeDf['unite_de_compte'] + ')'

    # Pivot crimeDf so each departement is a single row with crime stats as columns
    # We'll use both 'indicateur' and 'unite_de_compte' to build unique column names
    crimeDf['taux_pour_mille'] = crimeDf['taux_pour_mille'].astype(str).str.replace(',', '.').astype(float)
    crimeDf['nombre'] = crimeDf['nombre'].astype(int)
    crime_pivot = crimeDf.pivot(index='departement_code', columns='crime_col', values='taux_pour_mille')
    crime_pivot_nombre = crimeDf.pivot(index='departement_code', columns='crime_col', values='nombre')
    # Rename columns to indicate taux or nombre
    crime_pivot = crime_pivot.add_prefix('crime_taux_')
    crime_pivot_nombre = crime_pivot_nombre.add_prefix('crime_n_')
    # Merge both taux and nombre
    crime_pivot = crime_pivot.merge(crime_pivot_nombre, left_index=True, right_index=True)
    crime_pivot = crime_pivot.reset_index()


    # Merge with IAdf on departement_code
    IAdf = IAdf.merge(crime_pivot, left_on='department_code', right_on='departement_code', how='left')
    IAdf = IAdf.drop(columns=['departement_code'])
    del crimeDf, crime_pivot, crime_pivot_nombre  # Free resources

    departementsDf = pd.read_json(io.BytesIO(loadFile(minio, BUCKET_NAME, 'geo/departements.json'))).rename(columns={'code': 'department_code'})

    IAdf = IAdf.merge(
        departementsDf[['department_code', 'zone']],
        right_on='department_code',
        left_on='department_code',
        how='left',
        suffixes=('', '_geo')
    )

    IAdf = IAdf[(IAdf['zone'] == 'metro')]

    debugDF(IAdf)

    # Prepare the final DataFrame for IA-mart
    IAdf.to_sql(
        'ia_mart',
        iamart,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=1000
    )

    # Save IA-mart.csv to a buffer and upload to MinIO
    csv_buffer = io.BytesIO()
    IAdf.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
    csv_buffer.seek(0)
    minio.put_object(
        DATAMART_BUCKET_NAME,
        'IA-mart.csv',
        csv_buffer,
        length=csv_buffer.getbuffer().nbytes,
        content_type='text/csv; charset=utf-8'
    )
    print('IA-mart.csv uploaded to MinIO.')

if __name__ == "__main__":
    main()
