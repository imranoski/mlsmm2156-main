# local imports
from models import *


class EvalConfig:
    
    models = [
        ("content_based", ContentBased, {'features_method' : 'all' , 'regressor_method': 'linear'})  # model_name, model class, model parameters (dict)
    ]
    split_metrics = ["mae", "rmse"]
    #loo_metrics = ["hitrate"]
    #full_metrics = ["novelty"]
    loo_metrics = []
    full_metrics = []

    # Split parameters
    test_size = 0.25  # -- configure the test_size (from 0 to 1) --

    # Loo parameters
    top_n_value = 40  # -- configure the numer of recommendations (> 1) --
