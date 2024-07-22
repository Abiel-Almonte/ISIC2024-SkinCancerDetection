if __name__ == '__main___':
    from sklearn.model_selection import StratifiedGroupKFold
    from .util import ResUNetWithTabular, ResUNet,\
                    train_evaluate_model, oversample
    import warnings, pandas, yaml, os, torch, numpy

    warnings.filterwarnings('ignore')

    def load_config(fp):
        with open(fp, 'r') as file:
            return yaml.safe_load(file)
        
    def set_seed(seed: int= 42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config= load_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    metadata= pandas.read_csv(os.path.join(os.path.dirname(__file__), config['data']['metadata_file']))
    metadata= oversample(metadata)

    set_seed(config['seed'])

    splits= list(StratifiedGroupKFold(n_splits=2).split(metadata, metadata['target'], groups= metadata['patient_id']))

    if config['model']['name'] == 'ResUNetWithTabular':
        model = ResUNetWithTabular(config['model']['cont_features'], config['model']['bin_features'])
    elif config['model']['name'] == 'ResUNet':
        model = ResUNet()
    else:
        raise ValueError(f"Unsupported model: {config['model']['name']}")

    train_evaluate_model(metadata, config, splits, model)