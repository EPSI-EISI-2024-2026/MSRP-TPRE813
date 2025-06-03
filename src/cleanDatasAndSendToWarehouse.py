from clients.minio import get_minio_client
from clients.postgres import get_postgres_engine
import os
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

BUCKET_NAME = os.environ["MINIO_BUCKET"]

# Connect to MinIO
client = get_minio_client()

def loadFile(filename):
    """
    Load a file from MinIO bucket.
    """
    try:
        data = client.get_object(BUCKET_NAME, filename)
        return data.read()
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return None

def fileToDF(file, type: str):
    """
    Convert a file to a DataFrame.
    """
    if type == "xlsx":
        return pd.read_excel(BytesIO(file))
    elif type == "csv":
        return pd.read_csv(file)
    else:
        raise ValueError("Unsupported file type")

def clean_and_rename_columns(df):
    # Rename main columns except the first candidate columns
    rename_map = {
        'Code du département': 'departement_code',
        'Libellé du département': 'departement_name',
        'Etat saisie': 'entry_status',
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

    # Candidate fields to rename
    candidate_fields = [
        'Sexe', 'Nom', 'Prénom', 'Voix', '% Voix/Ins', '% Voix/Exp'
    ]
    # Find the index of the first candidate field
    first_candidate_idx = df.columns.get_loc(candidate_fields[0])
    # Rename the first set of candidate columns
    for j, field in enumerate(candidate_fields):
        col_idx = first_candidate_idx + j
        if col_idx < len(df.columns):
            df.columns.values[col_idx] = f"{field.lower().replace('% ', 'pct_').replace('/', '_').replace(' ', '_')}_candidat_1"
    # Prepare normalized field names for repeated candidates
    normalized_fields = [
        'sexe_candidat', 'nom_candidat', 'prenom_candidat',
        'voix_candidat', 'pct_voix_ins_candidat', 'pct_voix_exp_candidat'
    ]
    # Rename the repeated candidate columns
    candidate_num = 2
    for i in range(first_candidate_idx + len(candidate_fields), len(df.columns), len(candidate_fields)):
        for j, field in enumerate(normalized_fields):
            col_idx = i + j
            if col_idx < len(df.columns):
                df.columns.values[col_idx] = f"{field}_{candidate_num}"
        candidate_num += 1
    return df

def main():
    # Connect to Postgres
    # engine = get_postgres_engine()

    df = fileToDF(
        loadFile("elections-2022-depts-t1.xlsx"),
        "xlsx"
    )
    df = clean_and_rename_columns(df)
    print(df.columns)
    print("\nDataFrame extract:")
    print(df)
    

if __name__ == "__main__":
    main()
