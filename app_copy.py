import streamlit as st
import pandas as pd
from loaders import load_ratings, load_items
from models import SurpriseSVD, UserBased, ContentBased, Slope
from constants import Constant as C
import os

POSTERS_DIR = "posters"

def get_poster_path(movie_id):
    """Retourne le chemin du poster pour un film donné, ou None si absent."""
    if not os.path.exists(POSTERS_DIR):
        return None
    for fname in os.listdir(POSTERS_DIR):
        if str(movie_id) == fname.split("_")[0] or str(movie_id) == fname.split(".")[0]:
            return os.path.join(POSTERS_DIR, fname)
    return None


st.set_page_config(layout="wide")

# --- Gestion du rôle avec un bouton unique ---
if "role" not in st.session_state:
    st.session_state["role"] = "User"
role = st.session_state["role"]

# --- Header avec logo à gauche, titre au centre, bouton à droite ---
col_logo, col_title, col_btn = st.columns([1, 6, 1])
with col_logo:
    logo_path = "cherryflix.png"  # Assurez-vous que ce fichier existe dans le dossier du projet
    if os.path.exists(logo_path):
        st.image(logo_path, width=250)
    else:
        st.warning("Logo file not found.")
with col_title:
    st.markdown("<h1 style='text-align: center;'>Cherryflix</h1>", unsafe_allow_html=True)
with col_btn:
    if role == "User":
        if st.button("Admin mode"):
            st.session_state["role"] = "Admin"
            st.rerun()
    else:
        if st.button("User mode"):
            st.session_state["role"] = "User"
            st.rerun()

# Chargement des données
df_ratings = load_ratings()
df_items = load_items()
user_ids = sorted(df_ratings[C.USER_ID_COL].unique())
item_ids = sorted(df_items.index)

# --- Initialisation de l'User sélectionné ---
if "selected_user" not in st.session_state:
    st.session_state["selected_user"] = user_ids[0]  # ou autre valeur par défaut


# --- Admin view: model selection and training ---
if role == "Admin":
    model_name = st.selectbox(
        "Model",
        ["SurpriseSVD (SVD)", "UserBased", "ContentBased", "SlopeOne"]
    )

    if model_name == "ContentBased":
        features_method = st.selectbox("Features", ["all_limited","all"])
        regressor_method = st.selectbox("Regression type", ["linear_fi_true", "ridge_fi_true_alpha10.0"])
    
    # Selectbox User admin : met à jour la session explicitement
    selected_user = st.selectbox(
        "User :", user_ids, 
        index=user_ids.index(st.session_state["selected_user"]) if "selected_user" in st.session_state else 0,
    )
    if selected_user != st.session_state["selected_user"]:
        st.session_state["selected_user"] = selected_user

    # Entraînement du modèle (collaboratif uniquement)
    if model_name != "ContentBased":
        if st.button("Train model"):
            data = load_ratings(True)
            trainset = data.build_full_trainset()
            if model_name == "SurpriseSVD (SVD)":
                model = SurpriseSVD(n_factors=150)
            elif model_name == "UserBased":
                model = UserBased(k=50, sim_options={"name" : "jaccard", "min_support" : 2})
            elif model_name == "SlopeOne":
                model = Slope()
            model.fit(trainset)
            st.session_state["trained_model"] = model
            st.success("Model trained !")
else:
    # --- User view: pas de selectbox, utilise la session
    selected_user = st.session_state["selected_user"]
    model_name = "SurpriseSVD (SVD)"

# --- Prediction and display (shared) ---
if st.button("Compute recommendations for this user"):
    # Toujours utiliser l'User de la session
    selected_user = st.session_state["selected_user"]
    data = load_ratings(True)
    trainset = data.build_full_trainset()

    if model_name != "ContentBased":
        if "trained_model" not in st.session_state:
            st.warning("Model not trained (please train the model in admin mode).")
            st.stop()
        model = st.session_state["trained_model"]
    else:
        model = ContentBased(features_method=features_method, regressor_method=regressor_method)
        model.fit(trainset)
        st.session_state["trained_model"] = model  # <-- Add this line

    user_rated = set(df_ratings[df_ratings[C.USER_ID_COL] == selected_user][C.ITEM_ID_COL])
    items_in_trainset = set(trainset._raw2inner_id_items.keys())
    items_to_predict = [iid for iid in item_ids if iid not in user_rated and iid in items_in_trainset]

    testset = [(selected_user, iid, 0) for iid in items_to_predict]
    preds = model.test(testset)
    #df_preds = pd.DataFrame([{"userId": p.uid, "movieId": p.iid, "prediction": p.est} for p in preds])
    df_preds = pd.DataFrame([{"userId": p.uid, "movieId": p.iid} for p in preds])

    # Ajout du titre, genres, et visuel (supposé colonne 'image_url' dans df_items)
    if "image_url" in df_items.columns:
        df_preds = df_preds.merge(
            df_items[[C.LABEL_COL, C.GENRES_COL, "image_url"]],
            left_on="movieId", right_index=True, how="left"
        )
    else:
        df_preds = df_preds.merge(
            df_items[[C.LABEL_COL, C.GENRES_COL]],
            left_on="movieId", right_index=True, how="left"
        )
        df_preds["image_url"] = ""  # fallback si pas d'image

    st.session_state["df_preds"] = df_preds
    st.success("Recommendations computed !")

# Affichage
if "df_preds" in st.session_state:
    df_preds = st.session_state["df_preds"]
    df_preds = df_preds.reset_index(drop=True)

    all_genres = sorted({g for genres in df_preds[C.GENRES_COL].dropna() for g in genres})
    if "year" not in df_preds.columns:
        import re
        def extract_year(title):
            match = re.search(r"\((\d{4})\)", str(title))
            return int(match.group(1)) if match else None
        df_preds["year"] = df_preds[C.LABEL_COL].apply(extract_year)
    min_year = int(df_preds["year"].min())
    max_year = int(df_preds["year"].max())

    if role == "User":
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### Filters")
            selected_genres = st.multiselect("Genres :", all_genres)
            year_range = st.slider("Year :", min_year, max_year, (min_year, max_year))
        with col2:
            filtered = df_preds.copy()
            if selected_genres:
                filtered = filtered[filtered[C.GENRES_COL].apply(lambda genres: all(g in genres for g in selected_genres))]
            filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]
            st.write(f"Recommendations for user {selected_user} :")
            max_titles = 20
            n_cols = 5
            titles = filtered.head(max_titles)
            for i in range(0, len(titles), n_cols):
                cols = st.columns(n_cols)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(titles):
                        row = titles.iloc[idx]
                        # Affiche l'image si dispo, sinon un placeholder
                        placeholder_url = "https://via.placeholder.com/150x220?text=No+Image"
                        poster_path = get_poster_path(row["movieId"])
                        print(poster_path)
                        if poster_path and os.path.exists(poster_path):
                            col.image(poster_path, use_container_width=True, caption="")
                        else:
                            col.image(placeholder_url, use_container_width=True, caption="")

                        # Titre
                        col.markdown(f"<div style='text-align:center;font-weight:bold'>{row[C.LABEL_COL]}</div>", unsafe_allow_html=True)
                        # Genres en petit
                        genres_str = ", ".join(row[C.GENRES_COL]) if isinstance(row[C.GENRES_COL], list) else ""
                        col.markdown(f"<div style='text-align:center;font-size:12px;color:gray'>{genres_str}</div>", unsafe_allow_html=True)
    else:
        selected_genres = st.multiselect("Filter by genre :", all_genres)
        year_range = st.slider("Filter by year :", min_year, max_year, (min_year, max_year))
        filtered = df_preds.copy()
        if selected_genres:
            filtered = filtered[filtered[C.GENRES_COL].apply(lambda genres: all(g in genres for g in selected_genres))]
        filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]
        n_recos = st.slider("Number of displayed recommendations:", 1, min(50, len(filtered)), 10)
        st.write(f"Recommendations for user {selected_user} :")
        #st.dataframe(filtered[["movieId", C.LABEL_COL, C.GENRES_COL, "year", "prediction"]].head(n_recos))
        st.dataframe(filtered[["movieId", C.LABEL_COL, C.GENRES_COL, "year"]].head(n_recos))
        if model_name == "UserBased":
            st.markdown("#### Explanation of UserBased recommendation")
            # Limite la liste aux films affichés
            movie_ids = filtered.head(n_recos)["movieId"].tolist() if 'n_recos' in locals() else filtered["movieId"].tolist()
            movie_to_explain = st.selectbox(
                "Choose a movie to explain :",
                movie_ids,
                format_func=lambda x: df_items.loc[x, C.LABEL_COL]
            )
            if st.button("Explain recommendation for this movie"):
                model = st.session_state.get("trained_model", None)
                if model is not None:
                    explanation = model.explain(selected_user, movie_to_explain)
                    if explanation:
                        st.table(pd.DataFrame(explanation))
                    else:
                        st.info("No available explanation for this movie/user.")
                else:
                    st.warning("No available model for explanation.")

        if role == "Admin" and model_name == "ContentBased":
            if st.button("Explain recommendation for this user"):
                model = st.session_state.get("trained_model", None)  # <-- Add this line
                if model is not None:
                    explanation = model.explain(selected_user)
                    if explanation:
                        st.write("Feature importance for this user :")
                        st.json(explanation)
                    else:
                        st.info("No available explanation for this user.")
                else:
                    st.warning("No available model for explanation.")
else:
    st.info("Click on the button to generate recommendations.")