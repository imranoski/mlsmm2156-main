# standard library imports
from collections import defaultdict

# third parties imports
import numpy as np
import random as rd
import pandas as pd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD, PredictionImpossible
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from loaders import load_items, load_ratings, load_visuals, load_genome
from constants import Constant as C

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

'''
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


''' 
class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)

    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()
        df_ratings = load_ratings()
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

        elif features_method == "multi":
            mlb = MultiLabelBinarizer()
            df_genres = pd.DataFrame(mlb.fit_transform(df_items['genres']), columns=mlb.classes_, index=df_items.index)
            df_features = pd.concat([df_genres, df_items[C.YEAR]], axis=1)
            df_features.index = df_items.index 
            df_features = df_features.dropna()
            print(df_features)

        elif features_method == "multi_visual" : 
            mlb = MultiLabelBinarizer()
            df_visuals = load_visuals(mode = 'quantile')
            df_genres = pd.DataFrame(mlb.fit_transform(df_items['genres']), columns=mlb.classes_, index=df_items.index)
            df_features = pd.concat([df_genres, df_items[C.YEAR]], axis=1)
            df_features.index = df_items.index 
            df_features = df_features.merge(df_visuals, how = 'inner', left_index=True, right_index=True)
            df_features = df_features.dropna()
            print(df_features)

        elif features_method == "all": 
            mlb = MultiLabelBinarizer()
            df_visuals = load_visuals(mode = 'quantile')
            df_genome = load_genome()
            df_genres = pd.DataFrame(mlb.fit_transform(df_items['genres']), columns=mlb.classes_, index=df_items.index)
            df_genome = pd.DataFrame(mlb.fit_transform(df_genome[C.GENOME_TAG]), columns = mlb.classes_, index=df_genome.index)
            #df_features = pd.concat([df_genres, df_items[C.YEAR], df_genome], axis=1)
            df_features = df_items[[C.YEAR]].join([df_genres, df_genome, df_visuals], how='left').fillna(0)
            df_features.index = df_items.index 
            #df_features = df_features.merge(df_visuals, how = 'inner', left_index=True, right_index=True)
            #df_features = df_features.merge(df_genome, how = 'inner', on = C.ITEM_ID_COL)
            df_features = df_features.dropna()
            print(df_features)
            
        else: # (implement other feature creations here)
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        return df_features
    

    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)
        
        # Preallocate user profiles
        self.user_profile = {u: None for u in trainset.all_users()}

        if self.regressor_method == 'random_score':
            pass
        
        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]

            # (implement here the regressor fitting)  
        elif self.regressor_method == 'linear':
            for u in self.user_profile:
                ratings = self.trainset.ur[u]
                df_user = pd.DataFrame(ratings, columns=['inner_item_id', 'user_ratings'])
                df_user["item_id"] = df_user["inner_item_id"].map(self.trainset.to_raw_iid)
                df_user = df_user.merge(self.content_features, how='left', left_on='item_id', right_index=True )
                
                df_user = df_user.dropna()

                if len(df_user) == 0:
                    self.user_profile[u] = None
                    continue

                feature_names = list(self.content_features.columns)
                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = LinearRegression(fit_intercept=False)
                reg.fit(X, y)

                self.user_profile[u] = reg
        elif self.regressor_method == 'sgd':
            for u in self.user_profile:
                ratings = self.trainset.ur[u]
                df_user = pd.DataFrame(ratings, columns=['inner_item_id', 'user_ratings'])
                df_user["item_id"] = df_user["inner_item_id"].map(self.trainset.to_raw_iid)
                df_user = df_user.merge(self.content_features, how='left', left_on='item_id', right_index=True )
                
                df_user = df_user.dropna()

                if len(df_user) == 0:
                    self.user_profile[u] = None
                    continue

                feature_names = list(self.content_features.columns)
                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = SGDRegressor(fit_intercept=False)
                reg.fit(X, y)

                self.user_profile[u] = reg
        elif self.regressor_method == 'svr':
            for u in self.user_profile:
                ratings = self.trainset.ur[u]
                df_user = pd.DataFrame(ratings, columns=['inner_item_id', 'user_ratings'])
                df_user["item_id"] = df_user["inner_item_id"].map(self.trainset.to_raw_iid)
                df_user = df_user.merge(self.content_features, how='left', left_on='item_id', right_index=True )
                
                df_user = df_user.dropna()

                if len(df_user) == 0:
                    self.user_profile[u] = None
                    continue

                feature_names = list(self.content_features.columns)
                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = LinearSVR(fit_intercept=False)
                reg.fit(X, y)

                self.user_profile[u] = reg        
        elif self.regressor_method == 'random_forest':
            for u in self.user_profile:
                ratings = self.trainset.ur[u]
                df_user = pd.DataFrame(ratings, columns=['inner_item_id', 'user_ratings'])
                df_user["item_id"] = df_user["inner_item_id"].map(self.trainset.to_raw_iid)
                df_user = df_user.merge(self.content_features, how='left', left_on='item_id', right_index=True )
                
                df_user = df_user.dropna()

                if len(df_user) == 0:
                    self.user_profile[u] = None
                    continue

                feature_names = list(self.content_features.columns)
                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = RandomForestRegressor()
                reg.fit(X, y)

                self.user_profile[u] = reg    
        else:
                pass
        
    def estimate(self, u, i):
        trainset = self.trainset
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

        elif self.regressor_method in ['linear', 'sgd', 'logistic', 'svm', 'random_forest']:
            iid_raw = self.trainset.to_raw_iid(i)
            iid_int = int(iid_raw)
            if iid_int not in self.content_features.index:
                raise PredictionImpossible("Pas de features pour cet item")

            x = self.content_features.loc[iid_int].values.reshape(1, -1)
            score = self.predict(u, i)[0]
        else:
            score=None
            # (implement here the regressor prediction)

        return score

