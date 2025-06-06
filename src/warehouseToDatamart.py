import os
from dotenv import load_dotenv
from numpy import add
from clients.postgres import get_warehouse_engine, get_datamart_engine
import pandas as pd
from utils import debugDF, loadFile
from clients.minio import get_minio_client
import sqlalchemy
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# MinIO bucket name from .env
BUCKET_NAME = os.environ["MINIO_BUCKET"]
warehouse = get_warehouse_engine()
datamart = get_datamart_engine()
client = get_minio_client()

def addElectionTypeAndElection():
    # --- Handle Type_Election and Election in a transaction ---
    with datamart.begin() as conn:
        # Check if 'presidentielle' exists
        result = conn.execute(sqlalchemy.text("""
            SELECT id_type_election FROM type_election WHERE denomination = :denomination
        """), {"denomination": "presidentielle"})
        row = result.fetchone()
        if row:
            type_election_id = row[0]
            print(f"Type_Election 'presidentielle' exists with id {type_election_id}")
        else:
            # Insert and get id
            result = conn.execute(sqlalchemy.text("""
                INSERT INTO type_election (denomination) VALUES (:denomination) RETURNING id_type_election
            """), {"denomination": "presidentielle"})
            type_election_id = result.scalar()
            print(f"Type_Election 'presidentielle' created with id {type_election_id}")

        # --- Handle Election 2022 ---
        date_debut = '2022-04-10 00:00:00'
        date_fin = '2022-04-24 23:59:59'
        tour = 1
        result = conn.execute(sqlalchemy.text("""
            SELECT id_election FROM election WHERE date_debut = :date_debut AND date_fin = :date_fin AND tour = :tour AND id_type_election = :type_id
        """), {"date_debut": date_debut, "date_fin": date_fin, "tour": tour, "type_id": type_election_id})
        row = result.fetchone()
        if row:
            election_id = row[0]
            print(f"Election 2022 already exists with id {election_id}")
        else:
            result = conn.execute(sqlalchemy.text("""
                INSERT INTO election (date_debut, date_fin, tour, id_type_election) VALUES (:date_debut, :date_fin, :tour, :type_id) RETURNING id_election
            """), {"date_debut": date_debut, "date_fin": date_fin, "tour": tour, "type_id": type_election_id})
            election_id = result.scalar()
            print(f"Election 2022 created with id {election_id}")

def addGeoDatas():
    regionsDf = pd.read_json(BytesIO(loadFile(client, BUCKET_NAME, 'geo/regions.json')))
    departementsDf = pd.read_json(BytesIO(loadFile(client, BUCKET_NAME, 'geo/departements.json')))
    arrondissementsDf = pd.read_json(BytesIO(loadFile(client, BUCKET_NAME, 'geo/arrondissements.json')))
    communesDf = pd.read_json(BytesIO(loadFile(client, BUCKET_NAME, 'geo/communes.json')))

    # Drop typeLiaison column if present
    for df in [regionsDf, departementsDf, arrondissementsDf, communesDf]:
        if 'typeLiaison' in df.columns:
            df.drop(columns=['typeLiaison'], inplace=True)

    # Ensure codes are strings and strip whitespace
    regionsDf['code'] = regionsDf['code'].astype(str).str.strip()
    departementsDf['region'] = departementsDf['region'].astype(str).str.strip()

    # For each region, fetch the name of the chefLieu from communesDf and sum population
    chef_lieu_names = []
    region_populations = []
    for idx, region in regionsDf.iterrows():
        chef_lieu_code = region['chefLieu']
        commune_row = communesDf[communesDf['code'] == chef_lieu_code]
        if not commune_row.empty:
            chef_lieu_name = commune_row.iloc[0]['nom']
        else:
            chef_lieu_name = None
        chef_lieu_names.append(chef_lieu_name)
        communes_in_region = communesDf[communesDf['region'] == region['code']]
        total_population = communes_in_region['population'].fillna(0).sum()
        region_populations.append(int(total_population))
    regionsDf['chefLieu_nom'] = chef_lieu_names
    regionsDf['population'] = region_populations
    debugDF(regionsDf)

    # For each departement, fetch the name of the chefLieu from communesDf and sum population
    dep_chef_lieu_names = []
    dep_populations = []
    for idx, dep in departementsDf.iterrows():
        chef_lieu_code = dep['chefLieu']
        commune_row = communesDf[communesDf['code'] == chef_lieu_code]
        if not commune_row.empty:
            chef_lieu_name = commune_row.iloc[0]['nom']
        else:
            chef_lieu_name = None
        dep_chef_lieu_names.append(chef_lieu_name)
        communes_in_dep = communesDf[communesDf['departement'] == dep['code']]
        total_population = communes_in_dep['population'].fillna(0).sum()
        dep_populations.append(int(total_population))
    departementsDf['chefLieu_nom'] = dep_chef_lieu_names
    departementsDf['population'] = dep_populations
    debugDF(departementsDf)

    # Load the regions in db or update if exists and keep the ids later
    regionsDf_db = regionsDf.rename(columns={
        'nom': 'nom_region',
        'code': 'code_region',
        'chefLieu_nom': 'chef_lieu',
        'population': 'population'
    })[['nom_region', 'code_region', 'chef_lieu', 'population']]
    regionsDf_db.to_sql(
        'region',
        con=datamart,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    # Fetch region ids for mapping (reload after insert)
    region_id_map = pd.read_sql('SELECT id_region, code_region FROM region', datamart).set_index('code_region')['id_region'].to_dict()

    # Load the departements in db or update if exists and keep the ids later use the regions ids from earlier for proper relation
    departementsDf_db = departementsDf.rename(columns={
        'nom': 'nom_departement',
        'code': 'code_departement',
        'chefLieu_nom': 'chef_lieu',
        'population': 'population',
        'region': 'code_region'
    })[['nom_departement', 'code_departement', 'chef_lieu', 'population', 'code_region']]
    departementsDf_db['id_region'] = departementsDf_db['code_region'].map(region_id_map)
    departementsDf_db = departementsDf_db.drop(columns=['code_region'])
    departementsDf_db.to_sql(
        'departement',
        con=datamart,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    # Fetch departement ids for mapping
    dep_id_map = pd.read_sql('SELECT id_departement, code_departement FROM departement', datamart).set_index('code_departement')['id_departement'].to_dict()

    # Load communes in db or update if exists and keep the ids later use the departements ids from earlier for proper relation
    communesDf_db = communesDf.rename(columns={
        'code': 'code_insee',
        'nom': 'nom_commune',
        'population': 'population',
        'departement': 'code_departement',
    })[['code_insee', 'nom_commune', 'population', 'codesPostaux', 'code_departement']]
    # Assign the full array of postal codes
    communesDf_db['code_postal'] = communesDf_db['codesPostaux']
    communesDf_db['id_departement'] = communesDf_db['code_departement'].map(dep_id_map)
    communesDf_db = communesDf_db.drop(columns=['codesPostaux', 'code_departement'])
    communesDf_db.to_sql(
        'commune',
        con=datamart,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    print("Regions, departements, and communes loaded into datamart.")

    com_id_map = pd.read_sql('SELECT id_commune, code_insee FROM commune', datamart).set_index('code_insee')['id_commune'].to_dict()

    return com_id_map, dep_id_map, region_id_map

def extract_unique_candidates(electionDf):
    # Find all candidate columns by suffix
    max_candidate = 0
    for col in electionDf.columns:
        if col.startswith('candidate_firstname_'):
            num = int(col.split('_')[-1])
            if num > max_candidate:
                max_candidate = num
    # Build records for each candidate
    records = []
    for i in range(1, max_candidate + 1):
        first = f'candidate_firstname_{i}'
        last = f'candidate_lastname_{i}'
        gender = f'candidate_gender_{i}'
        if first in electionDf.columns and last in electionDf.columns and gender in electionDf.columns:
            subset = electionDf[[first, last, gender]].copy()
            subset.columns = ['candidate_firstname', 'candidate_lastname', 'candidate_gender']
            records.append(subset)
    all_candidates = pd.concat(records, ignore_index=True)
    unique_candidates = all_candidates.drop_duplicates().reset_index(drop=True)
    return unique_candidates

def loadCandidatesToDatamart(electionDf):
    unique_candidates = extract_unique_candidates(electionDf)

    # Map 'M' to 'H' for candidate_gender before renaming
    unique_candidates['candidate_gender'] = unique_candidates['candidate_gender'].replace({'M': 'H'})
    # Dump unique_candidates to the Candidats table in datamart
    unique_candidates = unique_candidates.rename(columns={
        'candidate_firstname': 'prenom_candidat',
        'candidate_lastname': 'nom_candidat',
        'candidate_gender': 'sexe_candidat',
    })
    unique_candidates['position_liste'] = None  # Add position_liste as NULL for now
    unique_candidates.to_sql(
        'candidats',
        con=datamart,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    print("unique_candidates DataFrame successfully dumped to Candidats table in datamart.")
    candidats_id_map = pd.read_sql('SELECT id_candidat, prenom_candidat, nom_candidat, sexe_candidat FROM candidats', datamart).set_index(['prenom_candidat', 'nom_candidat', 'sexe_candidat'])['id_candidat'].to_dict()
    print(f"Loaded {candidats_id_map} unique candidates into datamart.")

    return candidats_id_map

def insert_shared_votes(election_id, electionDf, dep_map):
    dep_col = 'department_code'
    shared_votes_rows = []
    for idx, row in electionDf.iterrows():
        dep_code = str(row[dep_col])
        id_departement = dep_map.get(dep_code)
        if id_departement is None:
            continue
        shared_votes_rows.append({
            'id_election': election_id,
            'id_departement': id_departement,
            'blanks_count': int(row['blank_votes']) if not pd.isna(row['blank_votes']) else 0,
            'blanks_prct': float(row['pct_blank_voters']) if not pd.isna(row['pct_blank_voters']) else 0.0,
            'nulls_count': int(row['null_votes']) if not pd.isna(row['null_votes']) else 0,
            'nulls_prct': float(row['pct_null_voters']) if not pd.isna(row['pct_null_voters']) else 0.0,
        })
    shared_votes_df = pd.DataFrame(shared_votes_rows)
    if not shared_votes_df.empty:
        shared_votes_df.to_sql(
            'shared_votes',
            con=datamart,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
        print(f"Inserted {len(shared_votes_df)} rows into Shared_Votes.")

def insert_votes(election_id, electionDf, dep_map, candidtas_map):
    dep_col = 'department_code'
    vote_rows = []
    for idx, row in electionDf.iterrows():
        dep_code = str(row[dep_col])
        id_departement = dep_map.get(dep_code)
        if id_departement is None:
            continue
        candidate_nums = [int(col.split('_')[-1]) for col in row.index if col.startswith('candidate_firstname_')]
        for num in candidate_nums:
            prenom = row.get(f'candidate_firstname_{num}')
            nom = row.get(f'candidate_lastname_{num}')
            sexe = row.get(f'candidate_gender_{num}')
            if pd.isna(prenom) or pd.isna(nom) or pd.isna(sexe):
                continue
            sexe = 'H' if sexe == 'M' else sexe
            id_candidat = candidtas_map.get((prenom, nom, sexe))
            if id_candidat is None:
                continue
            votes_count = row.get(f'candidate_votes_{num}', 0)
            valid_votes_prct = row.get(f'candidate_pct_votes_valid_{num}', 0.0)
            registered_votes_prct = row.get(f'candidate_pct_votes_registered_{num}', 0.0)
            vote_rows.append({
                'id_election': election_id,
                'id_candidat': id_candidat,
                'id_departement': id_departement,
                'votes_count': int(votes_count) if not pd.isna(votes_count) else 0,
                'valid_votes_prct': float(valid_votes_prct) if not pd.isna(valid_votes_prct) else 0.0,
                'registered_votes_prct': float(registered_votes_prct) if not pd.isna(registered_votes_prct) else 0.0,
            })
    votes_df = pd.DataFrame(vote_rows)
    if not votes_df.empty:
        votes_df.to_sql(
            'votes',
            con=datamart,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
        print(f"Inserted {len(votes_df)} rows into Votes.")

def sendChomageToDatamart(dep_map):
    # Read unemployment_rates from warehouse
    df = pd.read_sql_table('unemployment_rates', warehouse)
    # Use provided department id mapping
    df['id_departement'] = df['department_code'].map(dep_map)
    chomage_df = df[['id_departement', 'year', 'unemployment_rate']].rename(columns={'unemployment_rate': 'taux'})
    chomage_df = chomage_df.dropna(subset=['id_departement', 'year', 'taux'])
    chomage_df['taux'] = chomage_df['taux'].astype(float)
    chomage_df['year'] = chomage_df['year'].astype(str)
    chomage_df = chomage_df[['id_departement', 'year', 'taux']]
    chomage_df.to_sql(
        'chomage',
        con=datamart,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    print(f"Inserted {len(chomage_df)} rows into Chomage table in datamart.")

def sendCrimeDatas(dep_map):
    crimeDf = pd.read_sql_table('crime_data', warehouse)

    # Insert unique crime types and units into correct tables with only the 'label' column
    crimTypesDf = crimeDf['indicateur'].drop_duplicates().reset_index(drop=True).to_frame(name='label')
    crimTypesDf.to_sql(
        'crime_type',
        con=datamart,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    crimeUnitsDf = crimeDf['unite_de_compte'].drop_duplicates().reset_index(drop=True).to_frame(name='label')
    crimeUnitsDf.to_sql(
        'crime_unit',
        con=datamart,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    # Fetch ids for mapping
    crimTypesMap = pd.read_sql('SELECT id, label FROM crime_type', datamart).set_index('label')['id'].to_dict()
    crimeUnitsMap = pd.read_sql('SELECT id, label FROM crime_unit', datamart).set_index('label')['id'].to_dict()

    # Insert crime data into datamart with mapping from dep_map and crime types/units
    crimeDf['id_departement'] = crimeDf['Code_departement'].astype(str).map(dep_map)
    crimeDf['id_type'] = crimeDf['indicateur'].map(crimTypesMap)
    crimeDf['id_unit'] = crimeDf['unite_de_compte'].map(crimeUnitsMap)
    # Ensure correct types and column names
    crimes_to_insert = pd.DataFrame({
        'id_departement': crimeDf['id_departement'],
        'id_type': crimeDf['id_type'],
        'id_unit': crimeDf['id_unit'],
        'year': crimeDf['annee'].astype(str),
        'count': crimeDf['nombre'].astype(int),
        'taux_pour_mille': crimeDf['taux_pour_mille'].astype(str).str.replace(',', '.').astype(float),
        'population': crimeDf['insee_pop'].astype(int)
    })
    crimes_to_insert = crimes_to_insert.dropna(subset=['id_departement', 'id_type', 'id_unit', 'year', 'count', 'taux_pour_mille', 'population'])
    crimes_to_insert.to_sql(
        'crimes',
        con=datamart,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    print(f"Inserted {len(crimes_to_insert)} rows into Crimes table in datamart.")

def sendImmigrationDatas(dep_map):
    df = pd.read_sql_table('immigration_data', warehouse)
    df['id_departement'] = df['codgeo'].astype(str).map(dep_map)
    df = df.drop(columns=['codgeo', 'libgeo']).dropna().rename(columns={'an': 'year', 'part_immigres': 'taux'})

    debugDF(df)
    df.to_sql(
        'immigration',
        con=datamart,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )

def sendIncomeDatas(dep_map):
    df = pd.read_sql_table('income_data', warehouse)
    df['id_departement'] = df['code_dep'].astype(str).map(dep_map)
    df = df.drop(columns=['code_dep', 'lib_dep']).dropna().rename(columns={'Nombre': 'household_count', 'median': 'median_income'})

    debugDF(df)
    df.to_sql(
        'income',
        con=datamart,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )

def main():
    addElectionTypeAndElection()
    _, dep_map, _ = addGeoDatas()
    electionDf = pd.read_sql_table('elections_2022_departments', warehouse)
    candidtas_map = loadCandidatesToDatamart(electionDf)
    election_id = pd.read_sql('SELECT id_election FROM election WHERE date_debut = %s AND date_fin = %s', datamart, params=('2022-04-10 00:00:00', '2022-04-24 23:59:59')).iloc[0]['id_election']
    insert_shared_votes(election_id, electionDf, dep_map)
    insert_votes(election_id, electionDf, dep_map, candidtas_map)
    sendChomageToDatamart(dep_map)
    sendCrimeDatas(dep_map)
    sendImmigrationDatas(dep_map)
    sendIncomeDatas(dep_map)

if __name__ == "__main__":
    main()
