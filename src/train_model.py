from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class Trainer:
    def __init__(self, 
                 model: object,
                 dataset: dict,
                 **params):
        
        self.model = model # grid search object
        self.params = params 
        self.seed = params['seed']
        
        self.train_x = dataset['train_X']
        self.train_y = dataset['train_y']
        
        self.test_x = dataset['test_X']
        self.test_y = dataset['test_y']
        
        self.processor = dataset['processor'] # preprocessing  processor object
    
    def training(self):
        
        # define training pipeline
        train_pipeline = ImbPipeline(('processor', self.processor),
                                     ('smote', SMOTE(random_state = self.seed)),
                                     ('classifier', self.model))
        
        # fitting with train feature and train targets
        train_pipeline.fit(self.train_x, 
                           self.train_y)
        
        return train_pipeline