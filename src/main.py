from fire import Fire
from load_data import LoadDataset
from models.select_model import SelectModel
from train_model import Trainer
from test_model import Tester
from utils import load_model_config, logger, set_random_seed, log_param
import warnings
warnings.filterwarnings('ignore')


def main(params: dict):
    
    # step 0: initialization
    # setting the global seed number
    # it can fix the seed number globally
    set_random_seed(seed=params['seed'])
    log_param(params)
    logger.info('Initializing for Experiment')
    
    # step 1: load dataset and preprocessing
    dataset = LoadDataset(isTrain=True, params=params)
    processed_data = dataset.preprocessing() 
    logger.info('Loaded Train Datasets')
    
    # step 2: Load model
    model = SelectModel(model_name = params['model'], 
                        params=params)
    logger.info(f'Loaded model {params['model']}')
    
    # step 3: Train model and Hyper parameter tunning
    training = Trainer(model,
                       processed_data, 
                       params)
    
    logger.info(f'Start Model Training')
    tuned_model = training.model_train()
    logger.info(f'Training is Done!!')
    # it needs to report about result of model.....
    
    # # step 4: export best model and results
    testing = Tester(processed_data)
    logger.info('Start Test Phase')
    result = testing.model_test(tuned_model)
    logger.info(f'Testing is Done! The results are....')
    
    print(f'AUC: {result['auc']}\nBinary F1: {result['binary_f1']}\nMacro F1: {result['macro_f1']}\nMicro F1: {result['micro_f1']}')
    
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