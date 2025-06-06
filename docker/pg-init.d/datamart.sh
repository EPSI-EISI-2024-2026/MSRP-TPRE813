#!/bin/sh
set -e

# Use environment variables provided by the Postgres Docker image
DB_NAME="datamart"
DB_USER="${POSTGRES_USER:-postgres}"

# Create the datamart database if it doesn't exist
if ! psql -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    createdb -U "$DB_USER" "$DB_NAME"
fi

# Grant all privileges on the datamart database to the user
psql -v ON_ERROR_STOP=1 --username "$DB_USER" --dbname "$DB_NAME" <<-EOSQL
    GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO "$DB_USER";
    
    -- Create sexe_enum type if it does not exist
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'sexe_enum') THEN
            CREATE TYPE sexe_enum AS ENUM ('H', 'F');
        END IF;
    END$$;

    -- Create Candidats table if it does not exist
    CREATE TABLE IF NOT EXISTS Candidats(
        id_candidat SERIAL PRIMARY KEY,
        prenom_candidat VARCHAR(50),
        nom_candidat VARCHAR(50),
        sexe_candidat sexe_enum,
        position_liste INTEGER
    );

    CREATE TABLE IF NOT EXISTS Type_Election(
        id_type_election SERIAL,
        denomination VARCHAR(50) NOT NULL,
        PRIMARY KEY(id_type_election),
        UNIQUE(denomination)
    );

    CREATE TABLE IF NOT EXISTS Election(
        id_election SERIAL,
        date_debut TIMESTAMP NOT NULL,
        date_fin TIMESTAMP NOT NULL,
        tour INTEGER,
        id_type_election INTEGER NOT NULL,
        PRIMARY KEY(id_election),
        FOREIGN KEY(id_type_election) REFERENCES Type_Election(id_type_election)
    );

    CREATE TABLE IF NOT exists Region(
        id_region SERIAL,
        nom_region VARCHAR(50) NOT NULL,
        code_region VARCHAR(50) NOT NULL,
        chef_lieu VARCHAR(50),
        population INTEGER,
        PRIMARY KEY(id_region),
        UNIQUE(nom_region),
        UNIQUE(code_region)
    );

    CREATE TABLE IF NOT exists Departement(
        id_departement SERIAL,
        nom_departement VARCHAR(50) NOT NULL,
        code_departement VARCHAR(50) NOT NULL,
        chef_lieu VARCHAR(50),
        population INTEGER,
        id_region INTEGER NOT NULL,
        PRIMARY KEY(id_departement),
        UNIQUE(nom_departement),
        UNIQUE(code_departement),
        FOREIGN KEY(id_region) REFERENCES Region(id_region)
    );

    CREATE TABLE IF NOT exists Commune(
        id_commune SERIAL,
        code_insee VARCHAR(50),
        nom_commune VARCHAR(50),
        population VARCHAR(50),
        code_postal TEXT[],
        id_departement INTEGER NOT NULL,
        PRIMARY KEY(id_commune),
        FOREIGN KEY(id_departement) REFERENCES Departement(id_departement)
    );

    CREATE TABLE IF NOT EXISTS Shared_Votes(
        id_shared_vote SERIAL PRIMARY KEY,
        id_election INTEGER NOT NULL,
        id_departement INTEGER NOT NULL,
        blanks_count INTEGER NOT NULL,
        blanks_prct FLOAT NOT NULL,
        nulls_count INTEGER NOT NULL,
        nulls_prct FLOAT NOT NULL,
        FOREIGN KEY(id_election) REFERENCES Election(id_election),
        FOREIGN KEY(id_departement) REFERENCES Departement(id_departement)
    );

    CREATE TABLE IF NOT EXISTS Votes(
        id_vote SERIAL PRIMARY KEY,
        id_election INTEGER NOT NULL,
        id_candidat INTEGER NOT NULL,
        id_departement INTEGER NOT NULL,
        votes_count INTEGER NOT NULL,
        valid_votes_prct FLOAT NOT NULL,
        registered_votes_prct FLOAT NOT NULL,
        FOREIGN KEY(id_election) REFERENCES Election(id_election),
        FOREIGN KEY(id_candidat) REFERENCES Candidats(id_candidat),
        FOREIGN KEY(id_departement) REFERENCES Departement(id_departement)
    );

    create table if not exists Chomage(
        id SERIAL PRIMARY KEY,
        id_departement INTEGER NOT NULL,
        year VARCHAR(5) not null,
        taux FLOAT not null,
        FOREIGN KEY(id_departement) REFERENCES Departement(id_departement)
    );

    CREATE TABLE IF NOT EXISTS Crime_Type(
        id SERIAL PRIMARY KEY,
        label VARCHAR(100) NOT NULL
    );

    CREATE TABLE IF NOT EXISTS Crime_Unit(
        id SERIAL PRIMARY KEY,
        label VARCHAR(100) NOT NULL
    );

    CREATE TABLE IF NOT EXISTS Crimes(
        id SERIAL PRIMARY KEY,
        id_departement INTEGER NOT NULL,
        id_type INTEGER NOT NULL,
        id_unit INTEGER NOT NULL,
        year VARCHAR(5) NOT NULL,
        count INTEGER NOT NULL,
        taux_pour_mille FLOAT NOT NULL,
        population INTEGER NOT NULL,
        FOREIGN KEY(id_departement) REFERENCES Departement(id_departement),
        FOREIGN KEY(id_type) REFERENCES Crime_Type(id),
        FOREIGN KEY(id_unit) REFERENCES Crime_Unit(id)
    );

    CREATE TABLE IF NOT EXISTS Immigration(
        id SERIAL PRIMARY KEY,
        id_departement INTEGER NOT NULL,
        year VARCHAR(5) NOT NULL,
        taux FLOAT NOT NULL,
        FOREIGN KEY(id_departement) REFERENCES Departement(id_departement)
    );

    CREATE TABLE IF NOT EXISTS Income(
        id SERIAL PRIMARY KEY,
        id_departement INTEGER NOT NULL,
        household_count INTEGER NOT NULL,
        prct_taxed FLOAT NOT NULL,
        median_income INTEGER NOT NULL,
        FOREIGN KEY(id_departement) REFERENCES Departement(id_departement)
    );
EOSQL
