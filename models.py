# standard library imports
from collections import defaultdict

# third parties imports
import numpy as np
import random as rd
import pandas as pd
from surprise import AlgoBase
from surprise import KNNWithMeans, SlopeOne
from surprise import SVD
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from loaders import load_items, load_ratings, load_visuals, load_genome, load_items_tfidf
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone
from constants import Constant as C
import heapq
from surprise import PredictionImpossible, Prediction
from numpy import dot
from numpy.linalg import norm


regressor_map = {
            'linear_fi_true': LinearRegression(fit_intercept=True),
            'linear_fi_false': LinearRegression(fit_intercept=False),
            'sgd_fi_false': SGDRegressor(fit_intercept=False),
            'svr_fi_false': LinearSVR(fit_intercept=False),
            'sgd_fi_true': SGDRegressor(fit_intercept=True),
            'svr_fi_true': LinearSVR(fit_intercept=True),
            'ridge_fi_false' : Ridge(max_iter=1000, alpha=1.0, fit_intercept=False),
            'ridge_fi_true_alpha0.5' : Ridge(max_iter=1000, alpha=0.5, fit_intercept=True),
            'ridge_fi_true_alpha1.0' : Ridge(max_iter=1000, alpha=1.0, fit_intercept=True),
            'ridge_fi_true_alpha5.0' : Ridge(max_iter=1000, alpha=5.0, fit_intercept=True),
            'ridge_fi_true_alpha10.0' : Ridge(max_iter=1000, alpha=10.0, fit_intercept=True),

            'gradient' : GradientBoostingRegressor(n_estimators=100,  
                                                    learning_rate=0.1, 
                                                    max_depth=3,        
                                                    random_state=42)                            
        }

for i in range(10, 41):
    for j in range(1, 6):
        regressor_map[f'random_forest_est{i}_depth{j}'] = RandomForestRegressor(n_estimators=i, max_depth = j)

regressor_map = regressor_map


def get_top_n(predictions, n):
    """Return the top-N recommendation for each user from a set of predictions.
    Source: inspired by https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    and modified by cvandekerckh for random tie breaking

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    rd.seed(0)

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        rd.shuffle(user_ratings)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# Second algorithm
class ModelBaseline2(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        rd.seed(0)

    def estimate(self, u, i):
        return rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])


# Third algorithm
class ModelBaseline3(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        #self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])
        self.the_mean = np.mean([r for (_, _, r) in trainset.all_ratings()])
        return self

    def estimate(self, u, i):
        return self.the_mean


# Fourth Model
class SurpriseSVD(SVD):
    def __init__(self, n_factors=100, random_state = 42):
        SVD.__init__(self, n_factors=n_factors, random_state = random_state)

class Slope(SlopeOne):
    def __init__(self):
        SlopeOne.__init__(self)    


class KNNMeans(KNNWithMeans):
    def __init__(self, k=3, min_k=1, sim_options = {"name": "cosine","user_based": True}):  # compute  similarities between items)
        KNNWithMeans.__init__(self, k=k, min_k=min_k, sim_options=sim_options)

class UserBased(AlgoBase):
    def __init__(self, k=3, min_k=1, sim_options={}, **kwargs):
        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k

        
    def fit(self, trainset):
        self.user_profile = {u: None for u in trainset.all_users()}
        self.user_profile_explain = {}

        AlgoBase.fit(self, trainset)
        # -- implement here the fit function --
        self.compute_rating_matrix()
        self.mean_ratings = np.nanmean(self.ratings_matrix, axis=1)
        self.compute_similarity_matrix()
        return self

    def explain(self, u, i):
        """
        Retourne une explication de la prédiction UserBased pour un utilisateur et un film.
        Renvoie une liste de dicts : voisin, similarité, note du voisin pour ce film.
        """
        try:
            inner_uid = self.trainset.to_inner_uid(u)
            inner_iid = self.trainset.to_inner_iid(i)
        except ValueError:
            return []

        peergroup = []
        for (nb, rat) in self.trainset.ir[inner_iid]:
            if nb != inner_uid:
                sim = self.sim[inner_uid, nb]
                if sim > 0:
                    peergroup.append((nb, sim, rat))
        print("peergroup : ", peergroup)
        top_neighbours = heapq.nlargest(self.k, peergroup, key=lambda x: x[1])
        print("top neighbours : ", top_neighbours)
        explanation = []
        for nb, sim, rat in top_neighbours:
            raw_nb = self.trainset.to_raw_uid(nb)
            explanation.append({
                "neighbour": raw_nb,
                "similarity": round(float(sim), 3),
                "rating of the neighbour": float(rat)
            })
        return explanation
    
    def test(self, testset):
        predictions = []
        for uid, iid, rat in testset:
            try:
                est = self.estimate(uid, iid)
            except PredictionImpossible:
                est = self.trainset.global_mean  # ou continue si tu veux ignorer
            predictions.append(Prediction(uid, iid, rat, est, {}))
        return predictions
    

    def estimate(self, u, i):
        # Conversion sécurisée des IDs utilisateur/item
        try:
            inner_uid = self.trainset.to_inner_uid(u)
            inner_iid = self.trainset.to_inner_iid(i)
        except ValueError:
            raise PredictionImpossible('User and/or item is unknown.')

        est = self.mean_ratings[inner_uid]
        peergroup = []
        for (nb, rat) in self.trainset.ir[inner_iid]:
            if nb != inner_uid:
                sim = self.sim[inner_uid, nb]
                if sim > 0:
                    peergroup.append((nb, sim, rat))
        top_neighbours = heapq.nlargest(self.k, peergroup, key=lambda x: x[1])

        num = 0
        denom = 0
        actual_k = 0

        for (nb, sim, rat) in top_neighbours:
            num += sim * (rat - np.nanmean(self.ratings_matrix[nb]))
            denom += sim
            actual_k += 1

        if actual_k >= self.min_k and denom != 0:
            est = est + (num / denom)
            est = min(self.trainset.rating_scale[1], max(self.trainset.rating_scale[0], est))
            return est
        else:
            est = min(self.trainset.rating_scale[1], max(self.trainset.rating_scale[0], est))
            return est
                    
    def compute_rating_matrix(self):
        # -- implement here the compute_rating_matrix function --
        self.ratings_matrix = np.empty([self.trainset.n_users, self.trainset.n_items])
        self.ratings_matrix[:] = np.nan
        for uiid in self.trainset.ur:
            for iid, rating in self.trainset.ur[uiid]:
                self.ratings_matrix[uiid, iid] = rating
        return self.ratings_matrix
    
    def compute_similarity_matrix(self):
        # -- implement here the compute_rating_matrix function --
        self.sim = np.eye(self.trainset.n_users)
        similarity = 0
        for i in range(self.trainset.n_users):
            for j in range(i, self.trainset.n_users):
                # get common items
                common = ~np.isnan(self.ratings_matrix[i]) & ~np.isnan(self.ratings_matrix[j])
                if np.sum(common) >= self.sim_options['min_support']:
                    if self.sim_options['name'] == 'msd' :
                        diff = self.ratings_matrix[i][common] - self.ratings_matrix[j][common]
                        msd = np.mean(diff ** 2)
                        similarity = 1 / (1 + msd)
                    elif self.sim_options['name'] == 'jaccard':
                        union = np.count_nonzero(~np.isnan(self.ratings_matrix[i])) + np.count_nonzero(~np.isnan(self.ratings_matrix[j])) - np.count_nonzero(~np.isnan(self.ratings_matrix[i]) & ~np.isnan(self.ratings_matrix[j]))
                        similarity = np.sum(common) / union
                    elif self.sim_options['name'] == 'cosine':
                        ratings_i = self.ratings_matrix[i][common]
                        ratings_j = self.ratings_matrix[j][common]
                        if len(ratings_i) == 0 or norm(ratings_i) == 0 or norm(ratings_j) == 0:
                            similarity = 0
                        else:
                            similarity = dot(ratings_i, ratings_j) / (norm(ratings_i) * norm(ratings_j))
                    self.sim[i, j] = similarity
                    self.sim[j, i] = similarity
                
        print(self.sim)
        return self.sim


class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)

    def create_content_features(self, features_method):
        """Content Analyzer"""
        scaler = MinMaxScaler()
        df_items = load_items()
        df_items[[C.YEAR]] = scaler.fit_transform(df_items[[C.YEAR]])

        if features_method is None:
            df_features = None

        elif features_method == "title_length": # a naive method that creates only 1 feature based on title length
            df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')
            print(df_features)

        elif features_method == "all_limited" :
            df_genres = load_items_tfidf()
            df_features = df_items[[C.YEAR]].join(df_genres, how='left').fillna(0)
            print(df_features)

        elif features_method == "visual" :
            df_visuals = load_visuals(mode = 'quantile')
            df_visuals_ratings = df_items.merge(df_visuals, how = 'inner', left_index = True, right_index = True)
            df_features = df_visuals_ratings
            df_features = df_features.select_dtypes(include=[np.number])
            print(df_features)

        elif features_method == "all": 
            df_visuals = load_visuals(mode='quantile')
            df_genome = load_genome()
            df_genres = load_items_tfidf()

            overlap = set(df_genome.columns) & set(df_genres.columns)
            df_genres = df_genres.drop(columns=overlap)

            df_genome_renamed = df_genome.add_prefix('genome_')

            df_temp = pd.concat([df_genome_renamed, df_visuals], axis=1)

            df_features = df_items[[C.YEAR]].join([df_genres, df_temp], how='left').fillna(0)
            df_features.index = df_items.index

            df_features.to_csv('all-features.csv')
            print(df_features)

        else: # (implement other feature creations here)
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        print(df_features)
        df_features.to_csv('df_features.csv')
        return df_features

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.user_profile = {u: None for u in trainset.all_users()}
        self.user_profile_explain = {}

        if self.regressor_method not in regressor_map:
            print(f"Unsupported regressor: {self.regressor_method}")
            return

        model = regressor_map[self.regressor_method]

        for u in self.user_profile:
            ratings = self.trainset.ur[u]
            df_user = pd.DataFrame(ratings, columns=['inner_item_id', 'user_ratings'])
            df_user["item_id"] = df_user["inner_item_id"].map(self.trainset.to_raw_iid)

            # Merge features into df_user
            df_user = df_user.merge(self.content_features, how='left', left_on='item_id', right_index=True)
            df_user = df_user.dropna()

            if df_user.empty:
                self.user_profile[u] = None
                self.user_profile_explain[u] = None
                continue

            feature_names = list(self.content_features.columns)
            X = df_user[feature_names].values
            y = df_user["user_ratings"].values
            reg = clone(model)

            selector = SelectFromModel(estimator=reg, threshold='median')
            selector.fit(X, y)

            # Transform X
            X_selected = selector.transform(X)

            # Re-fit reg on selected features
            reg_selected = clone(model)
            reg_selected.fit(X_selected, y)

            # Store reg + selector
            self.user_profile[u] = {
                'model': reg_selected,
                'selector': selector
            }
            try:
                print(f'Intercept : {reg.intercept_}')
            except AttributeError:
                pass  


            selected_mask = selector.get_support()
            selected_features = X[:, selected_mask]

            weighted_features = np.average(selected_features, axis=0, weights=y)

            if weighted_features.sum() > 0:
                importance = weighted_features / weighted_features.sum()
            else:
                importance = np.zeros_like(weighted_features)

            full_scores = np.zeros(len(feature_names))
            full_scores[selected_mask] = importance 

            self.user_profile_explain[u] = dict(zip(feature_names, full_scores))

            print(f"Utilisateur {u} : y = {y}")
            print(f"Features shape : {X.shape}")
            print(f"Features sélectionnées : {selector.get_support().sum()}")
        
    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        # Vérifie que l'utilisateur et l'item existent
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        # Cas random_score
        if self.regressor_method == 'random_score':
            rd.seed()
            return rd.uniform(0.5, 5)

        # Récupère le profil utilisateur
        user_profile = self.user_profile.get(u)
        if user_profile is None:
            raise PredictionImpossible('No profile for this user.')

        # Cas régression classique
        if self.regressor_method in regressor_map:
            raw_item_id = self.trainset.to_raw_iid(i)
            if raw_item_id not in self.content_features.index:
                raise PredictionImpossible("no features for this item")

            model = user_profile['model']
            selector = user_profile['selector']
            x = self.content_features.loc[raw_item_id:raw_item_id, :].values  # (1, n_features)
            x_selected = selector.transform(x)
            print(f"User : {u} - Item {raw_item_id} - pred : {model.predict(x_selected)[0]}")
            return model.predict(x_selected)[0]

        # Si aucune méthode reconnue
        return None
    
    def explain(self, u):
        if u not in self.user_profile_explain or self.user_profile_explain[u] is None:
            return {}

        return self.user_profile_explain[u]
