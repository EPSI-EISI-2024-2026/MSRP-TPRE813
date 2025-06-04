import os
import re
from dotenv import load_dotenv
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

    # Clear tables before loading to avoid duplicates and FK issues
    with datamart.begin() as conn:
        conn.execute(sqlalchemy.text('DELETE FROM commune'))
        conn.execute(sqlalchemy.text('DELETE FROM departement'))
        conn.execute(sqlalchemy.text('DELETE FROM region'))

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
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=1000
    )
    print("unique_candidates DataFrame successfully dumped to Candidats table in datamart.")

def main():
    # addElectionTypeAndElection()
    addGeoDatas()

    # electionDf = pd.read_sql_table('elections_2022_departments', warehouse)
    # loadCandidatesToDatamart(electionDf)
    

if __name__ == "__main__":
    main()
