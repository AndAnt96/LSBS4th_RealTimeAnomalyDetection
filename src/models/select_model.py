from models.LIGHTGBM import LightGBM
from models.RANDOMFOREST import Randomforest
from models.XGBOOST import Xgboost
from models.VOTING import Voting

def SelectModel(model_name: str, 
                **params):
    
    # Every models are implement using the scikit-learn API
    model = {'lightgbm': LightGBM(params), 
             'randomforest': Randomforest(params), 
             'xgboost': Xgboost(params),
             'voting': Voting(params)}

    return model[model_name]