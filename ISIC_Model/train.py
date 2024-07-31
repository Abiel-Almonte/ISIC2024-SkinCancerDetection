from util import EfficientUNetWithTabular, EfficientUNet,\
     load_config, set_seed, prepare, train_evaluate_test, test_model
from arch_v2 import EfficientUNetWithTabular_v2
import warnings, pandas, os

warnings.filterwarnings('ignore')
config= load_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))
set_seed(config['seed'])

model_name= config['model']['name']
metadata= pandas.read_csv(os.path.join(
    os.path.dirname(__file__), config['data']['metadata_file'])
).drop(columns= ['Unnamed: 0']) #duplicate index

metadata, test_data= prepare(metadata, config['testing']['test_size'], config['seed'], _oversample=False)
train_data, valid_data= prepare(metadata, test_size=0.3, seed=config['seed'], _oversample= False)

if model_name== 'EfficientUNetWithTabular':
    model= EfficientUNetWithTabular_v2(config['model']['cont_features'], config['model']['bin_features'])
elif model_name== 'EfficientUNet':
    model= EfficientUNet()
else:
    raise ValueError(f"Unsupported model: {model_name}")
    
train_evaluate_args= {'train_data': train_data, 'valid_data': valid_data, 'config': config, 'model': model}
if config['testing']['test']:
    test_args= {'data': test_data, 'config': config, 'model': model}
else: test_args= None

if config['test_only']:
    pAUC= test_model(test_data, config, model)
else:
    pAUC= train_evaluate_test(train_evaluate_args, test_args)
if pAUC: print(f"pAUC: {pAUC:.3f}")