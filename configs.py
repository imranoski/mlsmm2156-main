# local imports
from models import *
from models import regressor_map

regressordict = regressor_map
print(regressordict)

class EvalConfig:
    
    sim_metrics = ["jaccard", "cosine", "msd"]
    knn_metrics = ["cosine", "msd"]
    ridge_alphas = [0.5, 1.0, 5.0, 10.0]

    content_models = ['linear_fi_true',
            'sgd_fi_true',
            'svr_fi_true',
            'ridge_fi_true',
            'gradient']

    models = [
        (f"slope", Slope, {})
    ]
    


    '''
    for i in content_models :
        models.append((f"content_based_{i}", ContentBased, {'features_method' : 'all' , 'regressor_method': i}))
    '''
    '''
    for i in ridge_alphas :
        models.append((f"content_based_ridge_alpha{i}", ContentBased, {'features_method' : 'all' , 'regressor_method':f'ridge_fi_true_alpha{i}'}))
    '''
    '''
    for i in regressordict.keys():
        if 'random_forest' in i :
            models.append((f"{i}", ContentBased, {'features_method' : 'all','regressor_method' : f'{i}'}))



    for i in range(11, 101, 2): 
        for m in range(1,10):
            for n in knn_metrics:
                    models.append(("knn_means_k"+str(i), KNNMeans, {'k': i, 'min_k' : 1, 'sim_options' : {"min_support": m, "name": n,"user_based": True}}))
 

    for i in range(100, 501, 100):
        models.append(("latent_"+str(i), SurpriseSVD, {'n_factors' : i}))

    for i in range(10, 51, 10) : 
        for k in range(1, 5):
            for j in sim_metrics:
                models.append((f"user_based_k{i}_sup_{k}_dist_{j}", UserBased, {'k': i, 'min_k' : 1, 'sim_options' : {"name": j, 'min_support' : 1}}))
    '''

    split_metrics = ["rmse"]
    #loo_metrics = ["hitrate"] 
    full_metrics = ["novelty"]
    loo_metrics = [] 
    #full_metrics = []


    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the number of recommendations (> 1) --
