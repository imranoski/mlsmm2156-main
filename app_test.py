# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from loaders import load_ratings, load_items
from models import (UserBased, ContentBased, ModelBaseline4)
from constants import Constant as C

# === INIT SESSION STATE ===
for key in ["role", "all_predictions", "admin_recos", "user_recos", "generate_recos", "user_generate_recos"]:
    if key not in st.session_state:
        st.session_state[key] = {} if 'recos' in key or 'predictions' in key else False
st.session_state.role = st.session_state.get("role", "Admin")

# === CHARGEMENT DES DONNEES ===
with st.spinner("Chargement des données..."):
    df_ratings = load_ratings()
    df_items = load_items()
    data = load_ratings(surprise_format=True)
    trainset = data.build_full_trainset()

# === UI - SWITCH ROLE ===
st.sidebar.title("🎭 Vue")
if st.sidebar.button(f"Passer en mode {'Utilisateur' if st.session_state.role == 'Admin' else 'Admin'}"):
    st.session_state.role = 'Utilisateur' if st.session_state.role == 'Admin' else 'Admin'
    st.stop()

# === ADMIN VIEW ===
if st.session_state.role == 'Admin':
    st.title("Système de recommandation - Admin")
    st.sidebar.header("Configuration")

    model_type = st.sidebar.selectbox("Type de modèle", ["Collaboratif", "Contenu", "Latent"], index=1)

    if model_type == "Collaboratif":
        sim_name = st.sidebar.selectbox("Mesure de similarité", ["msd", "jaccard", "cosine"])
        min_support = st.sidebar.slider("Support minimum", 1, 10, 3)
        k = st.sidebar.slider("k voisins", 1, 20, 3)
        min_k = st.sidebar.slider("min_k", 1, 5, 1)
    elif model_type == "Latent":
        n_factors = st.sidebar.slider("Nombre de facteurs", 1, 200, 100)
        random_state = 42
    else:
        features_method = st.sidebar.selectbox("Méthode features", ["title_length", "all_limited"])
        regressor_method = st.sidebar.selectbox("Régression", ["linear_fi_true", "random_forest", "ridge_fi_true"])

    selected_user = st.sidebar.selectbox("Utilisateur cible", df_ratings['userId'].unique())
    n_recommendations = st.sidebar.slider("Nb recommandations", 1, 100, 10)
    selected_year = st.sidebar.slider("Année", 1900, 2025, (1915, 2015))
    selected_genre = st.sidebar.selectbox("Genre", ["Tous"] + sorted({g for genres in df_items[C.GENRES_COL] for g in genres}))

    # === MODEL TRAINING ===
    model_params = (model_type, sim_name if model_type == "Collaboratif" else None,
                    features_method if model_type == "Contenu" else None,
                    regressor_method if model_type == "Contenu" else None,
                    n_factors if model_type == "Latent" else None)

    if st.session_state.get("model_params") != model_params:
        if model_type == "Collaboratif":
            model = UserBased(k=k, min_k=min_k, sim_options={'name': sim_name, 'min_support': min_support})
        elif model_type == "Latent":
            model = ModelBaseline4(n_factors=n_factors, random_state=random_state)
        else:
            model = ContentBased(features_method=features_method, regressor_method=regressor_method)

        model.fit(trainset)
        st.session_state.trained_model = model
        st.session_state.model_params = model_params
        st.session_state.all_predictions = {}
        st.session_state.admin_recos = {}

    model = st.session_state.trained_model

    if st.sidebar.button("Générer les recommandations"):
        all_items = set(df_items.index)
        user_rated = set(df_ratings[df_ratings['userId'] == selected_user]['movieId'])
        items_to_predict = list(all_items - user_rated)

        if selected_user not in st.session_state.all_predictions:
            testset = [(selected_user, iid, 0) for iid in items_to_predict]
            preds = model.test(testset)
            st.session_state.all_predictions[selected_user] = preds

        preds = st.session_state.all_predictions[selected_user]
        recs = []
        for pred in sorted(preds, key=lambda x: x.est, reverse=True):
            if pred.iid in df_items.index:
                item = df_items.loc[pred.iid]
                if selected_genre != "Tous" and selected_genre not in item[C.GENRES_COL]:
                    continue
                if not pd.isna(item[C.YEAR]) and selected_year[0] <= item[C.YEAR] <= selected_year[1]:
                    recs.append({"Titre": item['title'], "Année": item[C.YEAR],
                                 "Genres": ", ".join(item[C.GENRES_COL]), "Score": pred.est})
            if len(recs) >= n_recommendations:
                break

        st.session_state.admin_recos[selected_user] = pd.DataFrame(recs)

    df_rec = st.session_state.admin_recos.get(selected_user, pd.DataFrame())
    if not df_rec.empty:
        st.subheader(f"Top {n_recommendations} recommandations pour l'utilisateur {selected_user}")
        st.table(df_rec)

# === UTILISATEUR VIEW ===
else:
    st.title("🎬 Recommandations personnalisées")
    user_ids = df_ratings['userId'].unique()
    selected_user = st.selectbox("Sélectionnez votre identifiant", user_ids)

    st.sidebar.header("🎚️ Filtres")
    selected_years = st.sidebar.slider("Année", 1950, 2025, (1990, 2020))
    all_genres = sorted({genre for sublist in df_items[C.GENRES_COL] for genre in sublist})
    selected_genres = st.sidebar.multiselect("Genres", all_genres)
    n_recommendations = st.sidebar.slider("Nb recommandations", 5, 50, 10)

    model = st.session_state.get("trained_model")
    if not model:
        st.warning("Modèle non encore configuré par l'administrateur.")
    else:
        if selected_user not in st.session_state.all_predictions:
            rated = set(df_ratings[df_ratings['userId'] == selected_user]['movieId'])
            to_predict = list(set(df_items.index) - rated)
            preds = model.test([(selected_user, iid, 0) for iid in to_predict])
            st.session_state.all_predictions[selected_user] = preds

        preds = st.session_state.all_predictions[selected_user]
        recs = [(p.iid, p.est) for p in sorted(preds, key=lambda x: x.est, reverse=True)]

        filtered = []
        for mid, score in recs:
            if mid not in df_items.index:
                continue
            movie = df_items.loc[mid]
            if selected_years[0] <= movie[C.YEAR] <= selected_years[1]:
                if not selected_genres or set(movie[C.GENRES_COL]) & set(selected_genres):
                    filtered.append((mid, score))
            if len(filtered) >= n_recommendations:
                break

        st.subheader("🎥 Vos recommandations")
        if not filtered:
            st.info("Aucun film trouvé avec vos filtres.")
        else:
            for mid, score in filtered:
                movie = df_items.loc[mid]
                st.markdown(f"**{movie['title']}** ({movie[C.YEAR]})")
                st.markdown(f"Genres: {', '.join(movie[C.GENRES_COL])}")
                st.markdown(f"⭐ Score: {score:.2f}")
                st.markdown("---")
