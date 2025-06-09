import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CHARGER LES DONNÉES
print("📊 Chargement des données électorales...")
df = pd.read_csv("IA-mart.csv")

print(f"✅ Dataset chargé: {df.shape}")
print(f"Départements: {df['department_name'].nunique()}")

# 2. DÉFINIR LES BLOCS POLITIQUES (STRATÉGIE PRINCIPALE)
political_blocks = {
    'DROITE': ['MACRON', 'PECRESSE', 'LASSALLE'],
    'EXTREME_DROITE': ['LE PEN', 'ZEMMOUR'],
    'GAUCHE': ['MELENCHON', 'JADOT', 'HIDALGO', 'ROUSSEL'],
    'EXTREME_GAUCHE': ['ARTHAUD', 'POUTOU'],
    'AUTRES': ['DUPONT']
}

# Mapping inverse
candidate_to_block = {}
for block, candidates in political_blocks.items():
    for candidate in candidates:
        candidate_to_block[candidate] = block

print(f"\n🏛️ Blocs politiques définis:")
for block, candidates in political_blocks.items():
    print(f"  {block}: {', '.join(candidates)}")

# 3. COLONNES DE VOTES
vote_columns = [
    'votes_F-ARTHAUD-Nathalie',
    'votes_M-ROUSSEL-Fabien', 
    'votes_M-MACRON-Emmanuel',
    'votes_M-LASSALLE-Jean',
    'votes_F-LE PEN-Marine',
    'votes_M-ZEMMOUR-Eric',
    'votes_M-MELENCHON-Jean-Luc',
    'votes_F-HIDALGO-Anne',
    'votes_M-JADOT-Yannick',
    'votes_F-PECRESSE-Valerie',
    'votes_M-POUTOU-Philippe',
    'votes_M-DUPONT-AIGNAN-Nicolas'
]

# 4. VARIABLES PRÉDICTIVES
socio_economic_features = [
    'unemployment_rate',
    'median_income', 
    'households_count',
    'pct_taxed_households',
    'pct_immigrants',
    'pct_abstention',
    'pct_voters'
]

crime_features = [
    'crime_taux_coups_et_blessures_volontaires--(victime)',
    'crime_taux_cambriolages_de_logement--(infraction)',
    'crime_taux_escroqueries--(victime)',
    'crime_taux_trafic_de_stupefiants--(mis_en_cause)',
    'crime_taux_vols_de_vehicules--(vehicule)',
    'crime_taux_vols_dans_les_vehicules--(vehicule)'
]

all_features = socio_economic_features + crime_features

# 5. CRÉATION D'UN DATASET ENRICHI
def create_enhanced_individual_dataset(df, vote_cols, feature_cols, samples_per_dept=3000):
    """
    Crée un dataset avec des features enrichies et plus d'échantillons
    """
    individual_data = []
    
    for _, row in df.iterrows():
        dept_name = row['department_name']
        
        # Features de base
        dept_features = {}
        for col in feature_cols:
            if col in row and pd.notna(row[col]):
                dept_features[col] = row[col]
        
        # FEATURES ENRICHIES
        # 1. Ratios et interactions
        if 'median_income' in dept_features and 'unemployment_rate' in dept_features:
            dept_features['income_unemployment_ratio'] = dept_features['median_income'] / (dept_features['unemployment_rate'] + 0.1)
        
        # 2. Indice de criminalité global
        crime_cols = [col for col in feature_cols if 'crime_' in col]
        if crime_cols:
            crime_values = [dept_features.get(col, 0) for col in crime_cols]
            dept_features['crime_index'] = np.mean([v for v in crime_values if v > 0])
        
        # 3. Indice socio-économique
        if 'median_income' in dept_features and 'pct_taxed_households' in dept_features:
            dept_features['socio_economic_index'] = (
                dept_features['median_income'] / 30000 + 
                dept_features['pct_taxed_households'] / 100
            ) / 2
        
        # 4. Polarisation politique (écart type des votes)
        dept_votes = []
        candidate_votes = {}
        total_votes = 0
        
        for vote_col in vote_cols:
            if vote_col in df.columns and pd.notna(row[vote_col]):
                candidate_name = vote_col.replace('votes_', '').replace('F-', '').replace('M-', '').split('-')[0]
                votes = row[vote_col]
                candidate_votes[candidate_name] = votes
                dept_votes.append(votes)
                total_votes += votes
        
        if total_votes > 0:
            # Polarisation
            dept_features['political_polarization'] = np.std(dept_votes) / np.mean(dept_votes) if np.mean(dept_votes) > 0 else 0
            
            # Pour chaque candidat/bloc
            for candidate_name, votes in candidate_votes.items():
                if votes > 0:
                    proportion = votes / total_votes
                    
                    # Déterminer le bloc politique
                    political_block = candidate_to_block.get(candidate_name, 'AUTRES')
                    
                    # Nombre d'échantillons basé sur la proportion
                    base_samples = max(50, int(proportion * samples_per_dept))
                    
                    # Ajouter plus d'échantillons pour les blocs principaux
                    if political_block in ['DROITE', 'EXTREME_DROITE', 'GAUCHE']:
                        base_samples = int(base_samples * 1.5)
                    
                    # Créer les individus
                    for _ in range(base_samples):
                        individual = dept_features.copy()
                        
                        # Variations spécifiques par type de variable
                        for feat in individual:
                            if isinstance(individual[feat], (int, float)) and individual[feat] != 0:
                                if 'rate' in feat or 'pct' in feat:
                                    noise = np.random.normal(1, 0.05)
                                elif 'income' in feat:
                                    noise = np.random.normal(1, 0.12)
                                elif 'crime' in feat:
                                    noise = np.random.normal(1, 0.15)
                                else:
                                    noise = np.random.normal(1, 0.08)
                                individual[feat] = max(0, individual[feat] * noise)
                        
                        # Ajouter du bruit corrélé pour certaines variables
                        if 'unemployment_rate' in individual and 'median_income' in individual:
                            correlation_noise = np.random.normal(0, 0.02)
                            individual['unemployment_rate'] += correlation_noise
                            individual['median_income'] -= correlation_noise * 1000
                        
                        individual['candidat_vote'] = candidate_name
                        individual['bloc_politique'] = political_block
                        individual['department'] = dept_name
                        individual_data.append(individual)
    
    return pd.DataFrame(individual_data)

# 6. CRÉER LE DATASET ENRICHI
print("\n🔄 Création du dataset enrichi...")
df_individual = create_enhanced_individual_dataset(df, vote_columns, all_features)

# Ajouter les nouvelles features à la liste
enhanced_features = all_features + [
    'income_unemployment_ratio', 
    'crime_index', 
    'socio_economic_index', 
    'political_polarization'
]

print(f"✅ Dataset enrichi créé: {df_individual.shape}")
print(f"📊 Features totales: {len(enhanced_features)}")

# 7. ANALYSE DES DISTRIBUTIONS
print(f"\n📈 Distribution par candidat:")
candidate_counts = df_individual['candidat_vote'].value_counts()
for candidate, count in candidate_counts.items():
    pct = (count / len(df_individual)) * 100
    print(f"  {candidate}: {count:,} ({pct:.1f}%)")

print(f"\n🏛️ Distribution par bloc politique:")
bloc_counts = df_individual['bloc_politique'].value_counts()
for bloc, count in bloc_counts.items():
    pct = (count / len(df_individual)) * 100
    print(f"  {bloc}: {count:,} ({pct:.1f}%)")

# 8. STRATÉGIE MULTI-NIVEAU : MODÈLES SÉPARÉS

class HierarchicalElectoralPredictor:
    def __init__(self):
        self.bloc_model = None
        self.candidate_models = {}
        self.scaler = StandardScaler()
        self.enhanced_features = enhanced_features
        
    def fit(self, X, y_candidate, y_bloc):
        # Nettoyer les données
        X_clean = X[self.enhanced_features].fillna(X[self.enhanced_features].median())
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(X_clean.median())
        
        # Standardiser
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # 1. Modèle pour prédire le bloc politique
        print("🏛️ Entraînement du modèle de blocs politiques...")
        self.bloc_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
        )
        self.bloc_model.fit(X_scaled, y_bloc)
        
        # 2. Modèles spécialisés par bloc
        print("👥 Entraînement des modèles par bloc...")
        for bloc in political_blocks.keys():
            bloc_mask = y_bloc == bloc
            if bloc_mask.sum() > 50:  # Minimum d'échantillons
                X_bloc = X_scaled[bloc_mask]
                y_bloc_candidates = y_candidate[bloc_mask]
                
                if len(np.unique(y_bloc_candidates)) > 1:
                    model = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        random_state=42,
                        class_weight='balanced'
                    )
                    model.fit(X_bloc, y_bloc_candidates)
                    self.candidate_models[bloc] = model
                    print(f"  ✅ Modèle {bloc}: {len(y_bloc_candidates)} échantillons")
    
    def predict(self, X):
        X_clean = X[self.enhanced_features].fillna(X[self.enhanced_features].median())
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(X_clean.median())
        X_scaled = self.scaler.transform(X_clean)
        
        # Prédire les blocs
        predicted_blocs = self.bloc_model.predict(X_scaled)
        
        # Prédire les candidats par bloc
        predictions = []
        for i, bloc in enumerate(predicted_blocs):
            if bloc in self.candidate_models:
                candidate = self.candidate_models[bloc].predict(X_scaled[i:i+1])[0]
            else:
                # Fallback : candidat principal du bloc
                candidate = political_blocks[bloc][0]
            predictions.append(candidate)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        X_clean = X[self.enhanced_features].fillna(X[self.enhanced_features].median())
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(X_clean.median())
        X_scaled = self.scaler.transform(X_clean)
        
        # Probabilités des blocs
        bloc_probas = self.bloc_model.predict_proba(X_scaled)
        
        # Construire les probabilités finales
        all_candidates = sorted(df_individual['candidat_vote'].unique())
        final_probas = np.zeros((len(X), len(all_candidates)))
        
        for i in range(len(X)):
            for j, bloc in enumerate(self.bloc_model.classes_):
                bloc_prob = bloc_probas[i][j]
                
                if bloc in self.candidate_models and bloc_prob > 0:
                    candidate_probas = self.candidate_models[bloc].predict_proba(X_scaled[i:i+1])[0]
                    for k, candidate in enumerate(self.candidate_models[bloc].classes_):
                        if candidate in all_candidates:
                            candidate_idx = all_candidates.index(candidate)
                            final_probas[i][candidate_idx] += bloc_prob * candidate_probas[k]
                else:
                    # Distribuer uniformément sur les candidats du bloc
                    bloc_candidates = political_blocks.get(bloc, [bloc])
                    for candidate in bloc_candidates:
                        if candidate in all_candidates:
                            candidate_idx = all_candidates.index(candidate)
                            final_probas[i][candidate_idx] += bloc_prob / len(bloc_candidates)
        
        return final_probas

# 9. ENTRAÎNEMENT DU MODÈLE HIÉRARCHIQUE
print("\n🚀 Entraînement du modèle hiérarchique...")

# Préparer les données
X = df_individual[enhanced_features].copy()
y_candidate = df_individual['candidat_vote']
y_bloc = df_individual['bloc_politique']

# Split
X_train, X_test, y_candidate_train, y_candidate_test, y_bloc_train, y_bloc_test = train_test_split(
    X, y_candidate, y_bloc, test_size=0.2, stratify=y_bloc, random_state=42
)

# Créer et entraîner le modèle
hierarchical_model = HierarchicalElectoralPredictor()
hierarchical_model.fit(X_train, y_candidate_train, y_bloc_train)

# 10. ÉVALUATION
print("\n📊 Évaluation des performances...")

# Prédictions
y_pred_candidate = hierarchical_model.predict(X_test)
y_pred_bloc = hierarchical_model.bloc_model.predict(
    hierarchical_model.scaler.transform(
        X_test[enhanced_features].fillna(X_test[enhanced_features].median())
        .replace([np.inf, -np.inf], np.nan).fillna(X_test[enhanced_features].median())
    )
)

# Métriques
candidate_accuracy = accuracy_score(y_candidate_test, y_pred_candidate)
bloc_accuracy = accuracy_score(y_bloc_test, y_pred_bloc)

print(f"🎯 Accuracy candidats: {candidate_accuracy:.3f}")
print(f"🏛️ Accuracy blocs politiques: {bloc_accuracy:.3f}")

# Rapport détaillé
print(f"\n📋 Rapport de classification (candidats):")
print(classification_report(y_candidate_test, y_pred_candidate))

# 11. COMPARAISON AVEC MODÈLES CLASSIQUES
print("\n🆚 Comparaison avec modèles classiques...")

# Préparer les données pour modèles classiques
X_train_clean = X_train[enhanced_features].fillna(X_train[enhanced_features].median())
X_test_clean = X_test[enhanced_features].fillna(X_test[enhanced_features].median())

scaler_classic = StandardScaler()
X_train_scaled = scaler_classic.fit_transform(X_train_clean)
X_test_scaled = scaler_classic.transform(X_test_clean)

classic_models = {
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

classic_results = {}
for name, model in classic_models.items():
    try:
        if name == 'Random Forest':
            model.fit(X_train_clean, y_candidate_train)
            y_pred = model.predict(X_test_clean)
        else:
            model.fit(X_train_scaled, y_candidate_train)
            y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_candidate_test, y_pred)
        classic_results[name] = accuracy
        print(f"  {name}: {accuracy:.3f}")
        
    except Exception as e:
        print(f"  {name}: Erreur - {e}")

# 12. RÉSUMÉ FINAL
print(f"\n{'='*50}")
print(f"🏆 RÉSULTATS FINAUX")
print(f"{'='*50}")
print(f"Modèle hiérarchique (candidats): {candidate_accuracy:.3f}")
print(f"Modèle hiérarchique (blocs): {bloc_accuracy:.3f}")

if classic_results:
    best_classic = max(classic_results.items(), key=lambda x: x[1])
    print(f"Meilleur modèle classique: {best_classic[0]} ({best_classic[1]:.3f})")

# Vérifier si l'objectif est atteint
if candidate_accuracy >= 0.5:
    print(f"✅ OBJECTIF ATTEINT ! Accuracy ≥ 0.5")
else:
    print(f"⚠️ Objectif non atteint. Accuracy actuelle: {candidate_accuracy:.3f}")

# 13. FONCTION DE PRÉDICTION FINALE
def predict_vote_advanced(profil):
    """
    Fonction de prédiction avec le modèle hiérarchique
    """
    try:
        X_user = pd.DataFrame([profil])
        
        # Compléter les features manquantes
        for col in enhanced_features:
            if col not in X_user.columns:
                if col in X.columns:
                    X_user[col] = X[col].median()
                else:
                    X_user[col] = 0
        
        # Prédiction
        prediction = hierarchical_model.predict(X_user)[0]
        probabilities = hierarchical_model.predict_proba(X_user)[0]
        
        # Créer le dictionnaire des probabilités
        all_candidates = sorted(df_individual['candidat_vote'].unique())
        prob_dict = dict(zip(all_candidates, probabilities))
        
        return prediction, prob_dict
        
    except Exception as e:
        print(f"Erreur prédiction: {e}")
        return None, None

# 14. TEST FINAL
print(f"\n🎯 Test de prédiction avancée:")
profil_test = {
    'unemployment_rate': 8.0,
    'median_income': 25000,
    'pct_immigrants': 12.0,
    'pct_abstention': 28.0,
    'pct_taxed_households': 45.0
}

prediction, probabilities = predict_vote_advanced(profil_test)

if prediction and probabilities:
    print(f"Prédiction: {prediction}")
    print("Top 5 probabilités:")
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for candidate, prob in sorted_probs[:5]:
        bloc = candidate_to_block.get(candidate, 'AUTRES')
        print(f"  {candidate} ({bloc}): {prob:.1%}")

print(f"\n✅ Modèle prêt avec accuracy = {candidate_accuracy:.3f}")
