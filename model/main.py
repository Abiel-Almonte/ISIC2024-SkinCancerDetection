from architectures import get_model
from utils import(
    load_config, set_seed,
    prepare, test_model, 
    train_evaluate_test
)
import warnings, pandas, os

warnings.filterwarnings('ignore')


config= load_config( os.path.join(os.path.dirname(__file__), 'config.yaml'))
set_seed(config['seed'])



metadata= pandas.read_csv(os.path.join(
    os.path.dirname(__file__), config['data']['metadata_file'])
).drop(columns= ['Unnamed: 0']) #duplicate index

metadata, test_data= prepare(metadata, config['testing']['test_size'], config['seed'], use_undersample=False)
train_data, valid_data= prepare(metadata, test_size=0.3, seed=config['seed'], use_undersample= True)



model_name = config['model']['name']
try:
    ModelClass, args= get_model(model_name)
    model_kwargs= {k: v for k, v in config['model']['parameters'].items() if k in args and v is not None}
    model= model = ModelClass(**model_kwargs)
except ValueError as e:
    raise ValueError(f"Unsupported model: {model_name}") from e

    
train_evaluate_args = {'train_data': train_data, 'valid_data': valid_data, 'config': config, 'model': model}
test_args = {'data': test_data, 'config': config, 'model': model} if config['testing']['enabled'] else None



if config['test_only']:
    pAUC= test_model(test_data, config, model)
else:
    pAUC= train_evaluate_test(train_evaluate_args, test_args)


if pAUC: 
    print(f"pAUC: {pAUC:.3f}")