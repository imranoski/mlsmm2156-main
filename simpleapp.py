import streamlit as st
import pandas as pd
from loaders import load_ratings, load_items
from models import SurpriseSVD, UserBased, ContentBased, Slope
from constants import Constant as C

st.title("Recommandations - Démo simple")


# Chargement des données
df_ratings = load_ratings()
df_items = load_items()
user_ids = sorted(df_ratings[C.USER_ID_COL].unique())
item_ids = sorted(df_items.index)

# Sélection du modèle

model_name = st.selectbox(
    "Choisissez le modèle",
    ["SurpriseSVD (SVD)", "UserBased", "ContentBased", "SlopeOne"]
)

if model_name == "ContentBased":
    features_method = st.selectbox("Méthode features", ["title_length", "all_limited", "all"])
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

if st.button("Calculer les prédictions pour cet utilisateur"):
    data = load_ratings(True)
    trainset = data.build_full_trainset()

    # Pour les modèles collaboratifs, on réutilise le modèle entraîné
    if model_name != "ContentBased":
        if "trained_model" not in st.session_state:
            st.warning("Veuillez d'abord entraîner le modèle.")
            st.stop()
        model = st.session_state["trained_model"]
    else:
        # Pour ContentBased, on entraîne à chaque fois (profil utilisateur)
        model = ContentBased(features_method=features_method, regressor_method=regressor_method)
        model.fit(trainset)

    # Prédire uniquement pour les items non notés par l'utilisateur sélectionné
    user_rated = set(df_ratings[df_ratings[C.USER_ID_COL] == selected_user][C.ITEM_ID_COL])

    # Correction : ne garder que les items présents dans le trainset
    items_in_trainset = set(trainset._raw2inner_id_items.keys())
    items_to_predict = [iid for iid in item_ids if iid not in user_rated and iid in items_in_trainset]

    testset = [(selected_user, iid, 0) for iid in items_to_predict]
    preds = model.test(testset)
    df_preds = pd.DataFrame([{"userId": p.uid, "movieId": p.iid, "prediction": p.est} for p in preds])

    # Ajout du titre et des genres
    df_preds = df_preds.merge(
        df_items[[C.LABEL_COL, C.GENRES_COL]],
        left_on="movieId", right_index=True, how="left"
    )
    st.session_state["df_preds"] = df_preds
    st.session_state["last_model"] = model
    st.success("Prédictions calculées !")

# Affichage
if "df_preds" in st.session_state:
    df_preds = st.session_state["df_preds"]

    # Explication des recommandations (ContentBased uniquement)
    if model_name == "ContentBased":
        if st.button("Expliquer la recommandation pour cet utilisateur"):
            model = st.session_state.get("last_model", None)  # Récupération du modèle
            if model is not None:
                explanation = model.explain(selected_user)
                print(explanation)
                if explanation:
                    st.write("Importance des features pour cet utilisateur :")
                    df_exp = pd.DataFrame([{"feature" : f, "importance (%)" : round(i*100,2)} 
                                           for f, i in explanation.items()  if i != 0]).sort_values("importance (%)", ascending=False)
                    st.table(df_exp)
                else:
                    st.info("Aucune explication disponible pour cet utilisateur.")
            else:
                st.warning("Aucun modèle disponible pour l'explication.")

    # On trie par score de prédiction décroissant
    df_preds = df_preds.sort_values("prediction", ascending=False)
    df_preds = df_preds.reset_index(drop=True)

    # Widgets de filtrage
    # Filtre par genres (multi-sélection)
    all_genres = sorted({g for genres in df_preds[C.GENRES_COL].dropna() for g in genres})
    selected_genres = st.multiselect("Filtrer par genres :", all_genres)

    # Filtre par année (slider range)
    if "year" not in df_preds.columns:
        # Extraction de l'année depuis le titre si colonne absente
        import re
        def extract_year(title):
            match = re.search(r"\((\d{4})\)", str(title))
            return int(match.group(1)) if match else None
        df_preds["year"] = df_preds[C.LABEL_COL].apply(extract_year)

    min_year = int(df_preds["year"].min())
    max_year = int(df_preds["year"].max())
    year_range = st.slider("Filtrer par année :", min_year, max_year, (min_year, max_year))

    # Application des filtres
    if selected_genres:
        df_preds = df_preds[df_preds[C.GENRES_COL].apply(lambda genres: all(g in genres for g in selected_genres))]
    df_preds = df_preds[(df_preds["year"] >= year_range[0]) & (df_preds["year"] <= year_range[1])]

    # Slider pour limiter le nombre de recommandations affichées
    n_recos = st.slider("Nombre de recommandations à afficher :", 1, min(50, len(df_preds)), 10)
    st.write(f"Recommandations pour l'utilisateur {selected_user} (films non notés, filtrés) :")
    st.dataframe(df_preds[["movieId", C.LABEL_COL, C.GENRES_COL, "year", "prediction"]].head(n_recos))


else:
    st.info("Cliquez sur le bouton pour générer les prédictions.")

