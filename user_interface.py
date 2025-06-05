import streamlit as st
import pandas as pd
from loaders import load_ratings, load_items
from models import (UserBased, ContentBased, ModelBaseline4, get_top_n)
from constants import Constant as C
from surprise.model_selection import train_test_split
from pathlib import Path


st.title("Système de recommandation de films")
st.markdown("""
Ce système propose plusieurs algorithmes de recommandation :
- **Collaboratif** : Filtrage basé sur les utilisateurs similaires
- **Contenu** : Filtrage basé sur les caractéristiques des films
- **Latent** : Filtrage basé sur 
""")

# Charger les données
with st.spinner('Chargement des données...'):
    df_ratings = load_ratings()
    df_items = load_items()
    data = load_ratings(surprise_format=True)
    trainset, testset = train_test_split(data, test_size=0.2)
    #trainset = data.build_full_trainset()
    

# Sidebar pour la configuration
st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox(
    "Type de modèle",
    ["Collaboratif", "Contenu", "Latent"],
    index=1
)

# Sélection du modèle spécifique
if model_type == "Collaboratif":
    sim_name = st.sidebar.selectbox(
        "Mesure de similarité",
        ["msd", "jaccard", "cosine"]
    )
    min_support = st.sidebar.slider(
        "Support minimum pour similarité", 
        min_value=1, max_value=10, value=3
    )
    k = st.sidebar.slider(
        "Nombre de voisins (k)", 
        min_value=1, max_value=20, value=3
    )
    min_k = st.sidebar.slider(
        "Nombre minimum de voisins (min_k)", 
        min_value=1, max_value=5, value=1
    )

elif model_type == "Latent":
    n_factors = st.sidebar.slider(
        "Nombre de facteurs", 
        min_value=1, max_value=200, value=100)
    random_state = st.sidebar.slider(
        "Générateur de nombres aléatoires", 
        min_value=1, max_value=42, value=12)

else:  # Contenu
    if C.DATA_PATH == Path('data/hackathon'):
        features_method = st.sidebar.selectbox(
            "Méthode d'extraction de caractéristiques",
            ["title_length", "visual", "all"]
        )
    else :
        features_method = st.sidebar.selectbox(
            "Méthode d'extraction de caractéristiques",
            ["title_length", "all_limited"]
        )
    regressor_method = st.sidebar.selectbox(
        "Algorithme de régression",
        ["linear_fi_true", "linear_fi_false", 
            "sgd_fi_true", "sgd_fi_false",
            "svr_fi_true", "svr_fi_false",
            "random_forest", "ridge_fi_true",
            "ridge_fi_false", "gradient"]
    )

# Sélection de l'utilisateur
st.sidebar.header("Utilisateur")
user_ids = df_ratings['userId'].unique()
selected_user = st.sidebar.selectbox(
    "Choisissez un utilisateur :", 
    user_ids
)


# Nombre de recommandations
n_recommendations = st.sidebar.slider(
    "Nombre de recommandations", 
    min_value=1, max_value=100, value=10
)

# Bouton pour générer les recommandations
if st.sidebar.button("Générer les recommandations"):
    with st.spinner('Entraînement du modèle et génération des recommandations...'):
        try:
            if model_type == "Collaboratif":
                model = UserBased(
                    k=k,
                    min_k=min_k,
                    sim_options={
                        'name': sim_name,
                        'min_support': min_support
                    }
                )
            elif model_type == "Latent":
                model = ModelBaseline4(
                    n_factors=n_factors,
                    random_state=random_state
                )

            else:  
                model = ContentBased(
                    features_method=features_method,
                    regressor_method=regressor_method
                )

            # Entraînement du modèle
            model.fit(trainset)

            # Génération des recommandations
            all_items = set(df_items.index)
            user_rated_items = set(
                df_ratings[df_ratings['userId'] == selected_user]['movieId']
            )
            items_to_predict = list(all_items - user_rated_items)

            

            if model_type == "Collaboratif":
                known_items = []
                for iid in items_to_predict:
                    try:
                        _ = trainset.to_inner_iid(int(iid))
                        known_items.append(int(iid))
                    except ValueError:
                        continue  # On ignore les items inconnus du trainset

            else:  # Contenu
                # On garde tous les items à prédire
                known_items = items_to_predict

            
            testset = [(selected_user, iid, 0) for iid in known_items]
            predictions = model.test(testset)
            top_n = get_top_n(predictions, n=n_recommendations).get(selected_user, [])

            # Affichage des résultats
            st.subheader(f"Top {n_recommendations} recommandations pour l'utilisateur {selected_user}")
            
            if not top_n:
                st.warning("Aucune recommandation disponible pour cet utilisateur.")
            else:
                # Création d'un dataframe pour un affichage plus propre
                recommendations = []
                for movie_id, score in top_n:
                    title = df_items.loc[movie_id]['title'] if movie_id in df_items.index else str(movie_id)
                    year = df_items.loc[movie_id][C.YEAR] if movie_id in df_items.index else "Inconnu"
                    genres = ", ".join(df_items.loc[movie_id][C.GENRES_COL]) if movie_id in df_items.index else "Inconnu"
                    recommendations.append({
                        "Titre": title,
                        "Année": year,
                        "Genres": genres,
                        "Score": f"{score:.2f}"
                    })
                
                df_rec = pd.DataFrame(recommendations)
                st.table(df_rec)

                # Option pour exporter les résultats
                if st.button("Exporter les recommandations au format CSV"):
                    csv = df_rec.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name=f"recommandations_utilisateur_{selected_user}.csv",
                        mime='text/csv'
                    )

            # Section d'explication pour les modèles de contenu
            if model_type == "Contenu" and hasattr(model, 'explain'):
                st.subheader("Explication des recommandations")
                explanation = model.explain(trainset.to_inner_uid(selected_user))
                if explanation:
                    df_explanation = pd.DataFrame.from_dict(
                        explanation, 
                        orient='index', 
                        columns=['Importance']
                    ).sort_values('Importance', ascending=False)
                    st.write("Importance des caractéristiques pour cet utilisateur:")
                    st.dataframe(df_explanation.head(10))
                else:
                    st.info("Aucune explication disponible pour cet utilisateur.")

        except Exception as e:
            st.error(f"Une erreur est survenue : {str(e)}")

# Section pour afficher l'historique de l'utilisateur
st.header(f" Historique de l'utilisateur {selected_user}")
user_history = df_ratings[df_ratings['userId'] == selected_user]
user_history = user_history.merge(
    df_items, 
    left_on='movieId', 
    right_index=True
)

if not user_history.empty:
    # Tri par note décroissante
    user_history = user_history.sort_values('rating', ascending=False)
    
    # Affichage des films notés
    st.write(f"Nombre de films notés : {len(user_history)}")
    st.write("Films les mieux notés :")
    
    cols = st.columns(3)
    for idx, row in user_history.head(3).iterrows():
        with cols[idx % 3]:
            st.markdown(f"**{row['title']}** ({row[C.YEAR]})")
            st.markdown(f"Note: {row['rating']}/5")
            st.markdown(f"Genres: {', '.join(row[C.GENRES_COL])}")
else:
    st.warning("Cet utilisateur n'a pas encore noté de films.")

