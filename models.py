# standard library imports
from collections import defaultdict

# third parties imports
import numpy as np
import random as rd
import pandas as pd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD, PredictionImpossible
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from loaders import load_items, load_ratings, load_visuals, load_genome, load_items_tfidf
from sklearn.feature_selection import SelectFromModel
from sklearn.base import clone
from constants import Constant as C

regressor_map = {
            'linear_fi_true': LinearRegression(fit_intercept=True),
            'linear_fi_false': LinearRegression(fit_intercept=False),
            'sgd_fi_false': SGDRegressor(fit_intercept=False),
            'svr_fi_false': LinearSVR(fit_intercept=False),
            'sgd_fi_true': SGDRegressor(fit_intercept=True),
            'svr_fi_true': LinearSVR(fit_intercept=True),
            'random_forest': RandomForestRegressor(n_estimators=10, max_depth = 3),
            'ridge_fi_false' : Ridge(max_iter=1000, alpha=1.0, fit_intercept=False),
            'ridge_fi_true' : Ridge(max_iter=1000, alpha=1.0, fit_intercept=True),
            'gradient' : GradientBoostingRegressor(n_estimators=100,  
                                                    learning_rate=0.1, 
                                                    max_depth=3,        
                                                    random_state=42)                            
        }

linearmodels =    ['linear_fi_true',
            'linear_fi_false',
            'sgd_fi_false',
            'svr_fi_false',
            'sgd_fi_true',
            'svr_fi_true',
            'ridge_fi_true',
            'ridge_fi_false']

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


# First algorithm
class ModelBaseline1(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        return 2


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
class ModelBaseline4(SVD):
    def __init__(self):
        SVD.__init__(self, n_factors=100, random_state = 1)



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

            reg = selector.estimator_ 

            try:
                print(f'Intercept : {reg.intercept_}')
            except AttributeError:
                pass  

            self.user_profile[u] = {
                'model': reg,
                'selector': selector
            }

            weighted_features = np.average(X, axis=0, weights=y)

            if weighted_features.sum() > 0:
                importance = weighted_features / weighted_features.sum()
            else:
                importance = np.zeros_like(weighted_features)

            #print('Total of importance scores : ', importance.sum())

            full_scores = np.zeros(len(feature_names))
            full_scores[selector.get_support(indices=True)] = importance

            self.user_profile_explain[u] = dict(zip(feature_names, full_scores))
            

            print(self.user_profile_explain[u]) 
            '''
            Observations : 
            -- features : 'title_length', regressor = 'linear'
            RMSE when fit_intercept = False : 1.507315
            RMSE when fit_intercept = True : 1.08625

            In this context, the intercept can be interpreted as the average rating a user gives when the movie has no title.  
            When `fit_intercept=False`, the model assumes that ratings can take the value '0', which is not the case as the minimum rating is 0.5.

            '''
        
    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        # First, handle cases for unknown users and items
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')


        if self.regressor_method == 'random_score':
            rd.seed()
            score = rd.uniform(0.5,5)

        elif self.regressor_method == 'random_sample':
            rd.seed()
            score = rd.choice(self.user_profile[u])
        # (implement here the regressor prediction)
        if self.regressor_method in regressor_map:
            raw_item_id = self.trainset.to_raw_iid(i)
            
            if raw_item_id not in self.content_features.index:
                raise PredictionImpossible("no features for this item")

            x = self.content_features.loc[raw_item_id:raw_item_id, :].values

            user_profile = self.user_profile[u]
            model = user_profile['model']
            selector = user_profile['selector']

            x_selected = selector.transform(x)

            return model.predict(x_selected)[0]
        
        else:
            score=None

        return score
    
    def explain(self, u):
        if u not in self.user_profile_explain or self.user_profile_explain[u] is None:
            return {}

        return self.user_profile_explain[u]
