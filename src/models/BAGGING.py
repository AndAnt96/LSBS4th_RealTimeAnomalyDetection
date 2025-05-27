from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV

from utils import logger

def Bagging(params):
    
    param_set = {
        
    }
    
    classifier = BaggingClassifier(
        estimator=LGBMClassifier(random_state=42), 
        n_estimators=10,
        bootstrap=True,
        n_jobs=-1) 

    logger.info(f'load Bagging model')

    return {"parameters": param_set, 
            "estimator": classifier}