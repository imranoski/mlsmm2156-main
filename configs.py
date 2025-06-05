# local imports
from models import *


class EvalConfig:
    
    sim_metrics = ["jaccard", "cosine", "msd"]

    content_models = ['linear_fi_true',
            'sgd_fi_true',
            'svr_fi_true',
            'random_forest',
            'ridge_fi_true',
            'gradient']

    models = []
    

    for i in content_models :
        models.append((f"content_based_{i}", ContentBased, {'features_method' : 'all_limited' , 'regressor_method': i}))

    '''
    for i in range(10, 101, 10): 
        for j in range(1, 5):
            for m in range(1,10):
                for n in sim_metrics:
                    models.append(("knn_means_k"+str(i), KNNMeans, {'k': i, 'min_k' : j, 'sim_options' : {"min_support": m, "name": n,"user_based": True}}))
        continue
    '''   
    
    for i in range(10, 411, 10):
        models.append(("latent_"+str(i), SurpriseSVD, {'n_factors' : i}))

    for i in range(10, 51, 10) : 
        for k in range(1, 5):
            for j in sim_metrics:
                models.append((f"user_based_k{i}_sup_{k}_dist_{j}", UserBased, {'k': i, 'min_k' : 1, 'sim_options' : {"name": j, 'min_support' : 1}}))
    
    split_metrics = ["rmse"]
    #loo_metrics = ["hitrate"] 
    full_metrics = ["novelty"]
    loo_metrics = [] 
    #full_metrics = []


    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the number of recommendations (> 1) --
