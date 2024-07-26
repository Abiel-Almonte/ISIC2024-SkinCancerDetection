from util import EfficientUNetWithTabular, EfficientUNet, load_config,\
    set_seed, prepare, train_evaluate_test, test_model, oversample
from sklearn.model_selection import StratifiedGroupKFold
import warnings, pandas, os

warnings.filterwarnings('ignore')
config= load_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))
set_seed(config['seed'])

model_name= config['model']['name']
metadata= pandas.read_csv(os.path.join(os.path.dirname(__file__), config['data']['metadata_file'])).drop(columns= ['Unnamed: 0']) #duplicate index
metadata, metadata_test= prepare(metadata, config['testing']['test_size'], config['seed'])
splits= list(StratifiedGroupKFold(n_splits=2).split(metadata, metadata['target'], groups= metadata['patient_id']))

if model_name == 'ResUNetWithTabular':
    model = EfficientUNetWithTabular(config['model']['cont_features'], config['model']['bin_features'])
elif model_name == 'ResUNet':
    model = EfficientUNet()
else:
    raise ValueError(f"Unsupported model: {model_name}")
    
train_evaluate_args= {'data': metadata, 'config': config, 'splits': splits, 'model': model}
if config['testing']['test']:
    test_args= {'data': metadata_test, 'config': config, 'model': model}
else: test_args= None
#pAUC= test_model(metadata_test, config, model)
pAUC= train_evaluate_test(train_evaluate_args, test_args)
if pAUC:
    print(f"pAUC: {pAUC:.3f}")