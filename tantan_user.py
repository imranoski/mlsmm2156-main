import streamlit as st
import pandas as pd
from loaders import load_ratings, load_items
from models import SurpriseSVD, UserBased, ContentBased, Slope
from constants import Constant as C

st.set_page_config(layout="wide")

# --- Gestion du rôle avec un bouton unique ---
if "role" not in st.session_state:
    st.session_state["role"] = "Utilisateur"
role = st.session_state["role"]

# --- Header avec logo à gauche, titre au centre, bouton à droite ---
col_logo, col_title, col_btn = st.columns([1, 6, 1])
with col_logo:
    logo_url = "https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png"
    st.image(logo_url, width=60)
with col_title:
    st.markdown("<h1 style='text-align: center;'>goodBoyssss</h1>", unsafe_allow_html=True)
with col_btn:
    if role == "Utilisateur":
        if st.button("Passer en mode Admin"):
            st.session_state["role"] = "Admin"
            st.rerun()
    else:
        if st.button("Passer en mode Utilisateur"):
            st.session_state["role"] = "Utilisateur"
            st.rerun()

# Chargement des données
df_ratings = load_ratings()
df_items = load_items()
user_ids = sorted(df_ratings[C.USER_ID_COL].unique())
item_ids = sorted(df_items.index)

# --- Admin view: model selection and training ---
if role == "Admin":
    model_name = st.selectbox(
        "Choisissez le modèle",
        ["SurpriseSVD (SVD)", "UserBased", "ContentBased", "SlopeOne"]
    )

    if model_name == "ContentBased":
        features_method = st.selectbox("Méthode features", ["title_length", "all_limited"])
        regressor_method = st.selectbox("Régression", ["linear_fi_true", "random_forest", "ridge_fi_true"])

    selected_user = st.selectbox("Utilisateur :", user_ids)

    # Entraînement du modèle (collaboratif uniquement)
    if model_name != "ContentBased":
        if st.button("Entraîner le modèle"):
            data = load_ratings(True)
            trainset = data.build_full_trainset()
            if model_name == "SurpriseSVD (SVD)":
                model = SurpriseSVD(n_factors=50)
            elif model_name == "UserBased":
                model = UserBased(k=20, sim_options={"name" : "jaccard", "min_support" : 2})
            elif model_name == "SlopeOne":
                model = Slope()
            model.fit(trainset)
            st.session_state["trained_model"] = model
            st.success("Modèle entraîné !")
else:
    # --- User view: no user selection, use default user ---
    selected_user = user_ids[0]  # ou choisis un autre index si tu veux
    model_name = "SurpriseSVD (SVD)"
# --- Prediction and display (shared) ---
if st.button("Calculer les prédictions pour cet utilisateur"):
    data = load_ratings(True)
    trainset = data.build_full_trainset()

    if model_name != "ContentBased":
        if "trained_model" not in st.session_state:
            st.warning("Veuillez d'abord entraîner le modèle (Admin seulement).")
            st.stop()
        model = st.session_state["trained_model"]
    else:
        model = ContentBased(features_method=features_method, regressor_method=regressor_method)
        model.fit(trainset)

    user_rated = set(df_ratings[df_ratings[C.USER_ID_COL] == selected_user][C.ITEM_ID_COL])
    items_in_trainset = set(trainset._raw2inner_id_items.keys())
    items_to_predict = [iid for iid in item_ids if iid not in user_rated and iid in items_in_trainset]

    testset = [(selected_user, iid, 0) for iid in items_to_predict]
    preds = model.test(testset)
    df_preds = pd.DataFrame([{"userId": p.uid, "movieId": p.iid, "prediction": p.est} for p in preds])

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
    st.success("Prédictions calculées !")

# Affichage
if "df_preds" in st.session_state:
    df_preds = st.session_state["df_preds"]
    df_preds = df_preds.sort_values("prediction", ascending=False).reset_index(drop=True)

    all_genres = sorted({g for genres in df_preds[C.GENRES_COL].dropna() for g in genres})
    if "year" not in df_preds.columns:
        import re
        def extract_year(title):
            match = re.search(r"\((\d{4})\)", str(title))
            return int(match.group(1)) if match else None
        df_preds["year"] = df_preds[C.LABEL_COL].apply(extract_year)
    min_year = int(df_preds["year"].min())
    max_year = int(df_preds["year"].max())

    if role == "Utilisateur":
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### Filtres")
            selected_genres = st.multiselect("Genres :", all_genres)
            year_range = st.slider("Années :", min_year, max_year, (min_year, max_year))
        with col2:
            filtered = df_preds.copy()
            if selected_genres:
                filtered = filtered[filtered[C.GENRES_COL].apply(lambda genres: all(g in genres for g in selected_genres))]
            filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]
            st.write(f"Recommandations pour l'utilisateur {selected_user} :")
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
                        # ...existing code...
                        placeholder_url = "https://via.placeholder.com/150x220?text=No+Image"
                        if row["image_url"]:
                            col.image(row["image_url"], use_container_width=True, caption="")
                        else:
                            col.image(placeholder_url, use_container_width=True, caption="")

                        # Titre
                        col.markdown(f"<div style='text-align:center;font-weight:bold'>{row[C.LABEL_COL]}</div>", unsafe_allow_html=True)
                        # Genres en petit
                        genres_str = ", ".join(row[C.GENRES_COL]) if isinstance(row[C.GENRES_COL], list) else ""
                        col.markdown(f"<div style='text-align:center;font-size:12px;color:gray'>{genres_str}</div>", unsafe_allow_html=True)
    else:
        selected_genres = st.multiselect("Filtrer par genres :", all_genres)
        year_range = st.slider("Filtrer par année :", min_year, max_year, (min_year, max_year))
        filtered = df_preds.copy()
        if selected_genres:
            filtered = filtered[filtered[C.GENRES_COL].apply(lambda genres: all(g in genres for g in selected_genres))]
        filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]
        n_recos = st.slider("Nombre de recommandations à afficher :", 1, min(50, len(filtered)), 10)
        st.write(f"Recommandations pour l'utilisateur {selected_user} (films non notés, filtrés) :")
        st.dataframe(filtered[["movieId", C.LABEL_COL, C.GENRES_COL, "year", "prediction"]].head(n_recos))

        if role == "Admin" and model_name == "ContentBased":
            if st.button("Expliquer la recommandation pour cet utilisateur"):
                explanation = model.explain(selected_user)
                if explanation:
                    st.write("Importance des features pour cet utilisateur :")
                    st.json(explanation)
                else:
                    st.info("Aucune explication disponible pour cet utilisateur.")
else:
    st.info("Cliquez sur le bouton pour générer les prédictions.")