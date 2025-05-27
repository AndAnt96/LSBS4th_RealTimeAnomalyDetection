from fire import Fire
from load_data import LoadDataset
from models.select_model import SelectModel
from train_model import Trainer
from test_model import Tester
from utils import load_model_config, logger, set_random_seed

def main(model_name: str,
         params: dict):
    
    # step 0: initialization
    # setting the global seed number
    # it can fix the seed number globally
    set_random_seed(seed=params['seed'])
    
    # step 1: load dataset and data preprocessing
    dataset = LoadDataset(isTrain=True)
    processed_data = dataset.preprocessing() 
    logger.info('Loaded Train Datasets')
    
    # step 2: Model Selection
    model = SelectModel(model = model_name)
    logger.info(f'Loaded model {model_name}')
    
    # step 3: Train model and Hyper parameter tunning
    trained = Trainer(model, 
                      data= processed_data, 
                      **params)
    
    logger.info(f'Start Model Training')
    tuned_model = Trainer.parameter_searching()
    logger.info(f'Model Training is Done!!')
    # it needs to report about result of model.....
    
    # step 4: export best model and results
    dataset = LoadDataset(isTrain=False)
    processed_data = dataset.preprocessing()
    logger.info('Loaded Test Datasets') 
    
    testing = Tester(tuned_model)
    logger.info()

def main_wraper(model="voting", conf_file=None, **kwargs):
    base_config_path = f"../config/{model.lower()}.json"
    config = load_model_config(base_config_path)
    for k in kwargs:
        if k in config:
            logger.warning('{} will be overwritten!'.format(k))
            config[k] = kwargs[k]
        else:
            raise ValueError
    main(config)

if __name__ == '__main__':
    Fire(main_wraper)