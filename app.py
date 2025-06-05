import streamlit as st
import pandas as pd
from loaders import load_ratings, load_items
from models import (UserBased, ContentBased, ModelBaseline4, get_top_n)
from constants import Constant as C
from surprise.model_selection import train_test_split
from pathlib import Path


# Charger les données
with st.spinner('Chargement des données...'):
    df_ratings = load_ratings()
    df_items = load_items()
    data = load_ratings(surprise_format=True)
    trainset = data.build_full_trainset()

if 'all_predictions' not in st.session_state:
    st.session_state['all_predictions'] = {}
    
# === INIT ===
if 'role' not in st.session_state:
    st.session_state['role'] = 'Admin'

st.sidebar.title("🎭 Vue")
switch = st.sidebar.button(f"Passer en mode {'Utilisateur' if st.session_state['role'] == 'Admin' else 'Admin'}")
if switch:
    new_role = 'Utilisateur' if st.session_state['role'] == 'Admin' else 'Admin'
    st.session_state['role'] = new_role
    st.stop()  # Arrête ici pour éviter une exécution en double

# === ADMIN VIEW ===
if st.session_state['role'] == 'Admin':
    st.title("Système de recommandation de films - Administrateur")
    st.markdown("""
    Ce système propose plusieurs algorithmes de recommandation :
    - **Collaboratif** : Filtrage basé sur les utilisateurs similaires
    - **Contenu** : Filtrage basé sur les caractéristiques des films
    - **Latent** : Filtrage basé sur 
    """)

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
        random_state = 42

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

    # Filtres supplémentaires
    st.sidebar.header("Filtres avancés")
    selected_year = st.sidebar.slider(
        "Année", 1900, 2025, (1915, 2015))

    selected_genre = st.sidebar.selectbox(
        "Genre", ["Tous"] + sorted({g for genres in df_items[C.GENRES_COL] for g in genres})
    )

    # Initialisation du modèle dans la session si besoin
    model_params = (
        model_type,
        sim_name if model_type == "Collaboratif" else None,
        features_method if model_type == "Contenu" else None,
        regressor_method if model_type == "Contenu" else None,
        n_factors if model_type == "Latent" else None,
        random_state if model_type == "Latent" else None
    )

    # Entraînement uniquement si les hyperparamètres changent
    if 'trained_model' not in st.session_state or st.session_state.get('model_params') != model_params:
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
        model.fit(trainset)
        st.session_state['trained_model'] = model
        st.session_state['model_params'] = model_params
        st.session_state['admin_recos'] = {}  # reset recos si modèle change

    model = st.session_state['trained_model']

    # Bouton de génération des recommandations
    if 'generate_recos' not in st.session_state:
        st.session_state['generate_recos'] = False
    if 'admin_recos' not in st.session_state:
        st.session_state['admin_recos'] = {}

    def on_generate():
        st.session_state['generate_recos'] = True

    st.sidebar.button("Générer les recommandations", on_click=on_generate, key="sidebar_generate")

    # Génère uniquement si bouton cliqué (plus de recalcul automatique par utilisateur)
    if st.session_state['generate_recos']:
        with st.spinner('Génération des recommandations...'):
            try:
                all_items = set(df_items.index)
                user_rated_items = set(df_ratings[df_ratings['userId'] == selected_user]['movieId'])
                items_to_predict = list(all_items - user_rated_items)

                if model_type == "Collaboratif":
                    known_items = []
                    for iid in items_to_predict:
                        try:
                            _ = trainset.to_inner_iid(int(iid))
                            known_items.append(int(iid))
                        except ValueError:
                            continue
                else:
                    known_items = items_to_predict

                # Si prédictions pas encore faites pour ce modèle, les faire une seule fois
                if not st.session_state['all_predictions']:
                    full_testset = [(u, iid, 0) 
                                    for u in user_ids 
                                    for iid in df_items.index 
                                    if iid not in set(df_ratings[df_ratings['userId'] == u]['movieId'])]
                    predictions = model.test(full_testset)
                    for pred in predictions:
                        uid = pred.uid
                        if uid not in st.session_state['all_predictions']:
                            st.session_state['all_predictions'][uid] = []
                        st.session_state['all_predictions'][uid].append(pred)

                # Récupérer les prédictions pour l’utilisateur sélectionné
                user_preds = st.session_state['all_predictions'].get(selected_user, [])

                user_preds.sort(key=lambda x: x.est, reverse=True)

                recs = []
                for pred in user_preds:
                    movie_id = pred.iid
                    score = pred.est
                    if movie_id not in df_items.index:
                        continue
                    item = df_items.loc[movie_id]
                    year = item[C.YEAR]
                    genres = item[C.GENRES_COL]
                    recs.append({
                        "Titre": item['title'],
                        "Année": year,
                        "Genres": ", ".join(genres),
                        "Genres_list": genres,
                        "Score": score
                    })

                # Stockage en mémoire
                st.session_state['admin_recos'][selected_user] = pd.DataFrame(recs)

            except Exception as e:
                st.error(f"Erreur lors de la génération : {str(e)}")
        # Reset du flag
        st.session_state['generate_recos'] = False

    # === Affichage dynamique sans recalcul ===
    df_rec = st.session_state['admin_recos'].get(selected_user, pd.DataFrame())
    if not df_rec.empty:
        min_year, max_year = selected_year
        df_rec = df_rec[df_rec['Année'].apply(lambda y: not pd.isna(y) and min_year <= int(y) <= max_year)]

        if selected_genre != "Tous":
            df_rec = df_rec[df_rec['Genres_list'].apply(lambda genres: selected_genre in genres)]

        df_rec = df_rec.sort_values('Score', ascending=False).head(n_recommendations)

        st.subheader(f"Top {n_recommendations} recommandations pour l'utilisateur {selected_user}")
        if df_rec.empty:
            st.warning("Aucune recommandation avec ces filtres.")
        else:
            st.table(df_rec.drop(columns="Genres_list"))
            if st.button("Exporter les recommandations au format CSV", key="export_csv"):
                csv = df_rec.drop(columns="Genres_list").to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger CSV",
                    data=csv,
                    file_name=f"recommandations_utilisateur_{selected_user}.csv",
                    mime='text/csv'
                )
    else:
        st.info("Aucune recommandation disponible. Cliquez sur 'Générer les recommandations'.")

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


# === USER VIEW ===
else:
    st.title("🎬 Recommandations personnalisées")

    user_ids = df_ratings['userId'].unique()
    selected_user = st.selectbox("Sélectionnez votre identifiant :", user_ids)

    # Filtres
    st.sidebar.header("🎚️ Filtres")
    available_years = sorted(df_items[C.YEAR].dropna().unique())
    selected_years = st.sidebar.slider("Année de sortie", int(min(available_years)), int(max(available_years)), (1990, 2020))
    all_genres = sorted({genre for sublist in df_items[C.GENRES_COL] for genre in sublist})
    selected_genres = st.sidebar.multiselect("Genres", all_genres, default=[])

    n_recommendations = st.sidebar.slider("Nombre de films à afficher", 5, 50, 15)

    # Bouton pour générer les recommandations utilisateur
    if 'user_generate_recos' not in st.session_state:
        st.session_state['user_generate_recos'] = False
    if 'user_recos' not in st.session_state:
        st.session_state['user_recos'] = {}

    def on_user_generate():
        st.session_state['user_generate_recos'] = True

    st.sidebar.button("Générer mes recommandations", on_click=on_user_generate, key="user_generate")

    if 'trained_model' not in st.session_state:
        st.warning("Le modèle n'est pas encore configuré par l'administrateur.")
    else:
        # Génère les recommandations uniquement au clic
        if st.session_state['user_generate_recos'] or selected_user not in st.session_state['user_recos']:
            model = st.session_state['trained_model']
            all_items = set(df_items.index)
            rated_items = set(df_ratings[df_ratings['userId'] == selected_user]['movieId'])
            items_to_predict = list(all_items - rated_items)

            # Calculer toutes les prédictions une fois si non fait
            if not st.session_state['all_predictions']:
                full_testset = [(u, iid, 0) 
                                for u in user_ids 
                                for iid in df_items.index 
                                if iid not in set(df_ratings[df_ratings['userId'] == u]['movieId'])]
                predictions = model.test(full_testset)
                for pred in predictions:
                    uid = pred.uid
                    if uid not in st.session_state['all_predictions']:
                        st.session_state['all_predictions'][uid] = []
                    st.session_state['all_predictions'][uid].append(pred)

            # Utiliser les prédictions en mémoire
            user_preds = st.session_state['all_predictions'].get(selected_user, [])
            user_preds.sort(key=lambda x: x.est, reverse=True)
            top_n = [(p.iid, p.est) for p in user_preds][:100]

            st.session_state['user_recos'][selected_user] = top_n
            st.session_state['user_generate_recos'] = False

        # Filtrage dynamique sur les recommandations déjà calculées
        top_n = st.session_state['user_recos'].get(selected_user, [])

        # Filtrage par année et genre
        filtered = []
        for movie_id, score in top_n:
            if movie_id not in df_items.index:
                continue
            movie = df_items.loc[movie_id]
            if not (selected_years[0] <= movie[C.YEAR] <= selected_years[1]):
                continue

            # Filtre genres uniquement si au moins un genre est sélectionné
            if selected_genres:
                if not set(movie[C.GENRES_COL]).intersection(set(selected_genres)):
                    continue

            filtered.append((movie_id, score))
            if len(filtered) >= n_recommendations:
                break

        st.subheader("🎥 Vos recommandations")
        if not filtered:
            st.info("Aucun film ne correspond à vos filtres.")
        else:
            rows = [filtered[i:i+5] for i in range(0, len(filtered), 5)]
            for row in rows:
                cols = st.columns(len(row))
                for idx, (movie_id, score) in enumerate(row):
                    movie = df_items.loc[movie_id]
                    with cols[idx]:
                        st.markdown(f"**{movie['title']}** ({movie[C.YEAR]})")
                        st.markdown(f"Genres : {', '.join(movie[C.GENRES_COL])}")
                        st.markdown(f"⭐ Score : {score:.2f}")