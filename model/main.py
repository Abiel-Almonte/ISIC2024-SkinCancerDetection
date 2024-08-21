import wandb
from architectures import get_model
from utils import(
    load_config, set_seed,
    prepare, test_model, 
    train_evaluate_test
)
import warnings, pandas, os

def main():
    warnings.filterwarnings('ignore')


    config= load_config( os.path.join(os.path.dirname(__file__), 'config.yaml'))
    set_seed(config['seed'])

    wandb.init(
        project= "kaggle",
        entity= "almonteabiel-florida-international-university",
        config= config
    )

    metadata= pandas.read_csv(os.path.join(
        os.path.dirname(__file__), wandb.config['data']['metadata_file'])
    ).drop(columns= ['Unnamed: 0']) #duplicate index

    metadata, test_data= prepare(metadata, wandb.config['testing']['test_size'], wandb.config['seed'], use_undersample=False)
    train_data, valid_data= prepare(metadata, test_size= 0.3, seed= wandb.config['seed'], use_undersample= True)



    model_name = wandb.config['model']['name']
    try:
        ModelClass, args= get_model(model_name)
        model_kwargs= {k: v for k, v in wandb.config['model']['parameters'].items() if k in args and v is not None}
        model= ModelClass(**model_kwargs)
    except ValueError as e:
        raise ValueError(f"Unsupported model: {model_name}") from e

        
    train_evaluate_args = {'train_data': train_data, 'valid_data': valid_data, 'config': wandb.config, 'model': model}
    test_args = {'data': test_data, 'config': wandb.config, 'model': model} if wandb.config['testing']['enabled'] else None



    if wandb.config['test_only']:
        pAUC= test_model(test_data, wandb.config, model)
    else:
        pAUC= train_evaluate_test(train_evaluate_args, test_args)


    if pAUC: 
        print(f"pAUC: {pAUC:.3f}")

if __name__ == '__main__':
    sweep= load_config( os.path.join(os.path.dirname(__file__), 'sweep.yaml'))
    sweep_id =  wandb.sweep(sweep)

    wandb.agent(sweep_id=sweep_id, function=main, entity= "almonteabiel-florida-international-university", project= "kaggle")