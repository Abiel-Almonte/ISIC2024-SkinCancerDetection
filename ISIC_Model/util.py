import torch.utils
import torch
from arch import ResUNet, ResUNetWithTabular
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
from typing import Tuple, Any, Union, List, Dict, Callable
from sklearn.metrics import roc_auc_score
import pandas, numpy, fnmatch, os, gc, yaml, random, json
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(480),
    transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TRANSFORM = transforms.Compose([
    transforms.Resize(480),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SkinDataset(Dataset):
    def __init__(self, df:pandas.Series, transform, image_dir:str) -> None:
        super().__init__()
        self.df= df
        self.transform= transform
        self.image_dir= os.path.join(os.path.dirname(__file__), image_dir)
        
        self.df_cont= self.df.drop(columns=['isic_id', 'target', 'patient_id', 'sex', 'anterior torso',
                            'head/neck', 'lower extremity', 'posterior torso', 'upper extremity'])
        self.df_bin= self.df[['sex', 'anterior torso', 'head/neck', 'lower extremity', 'posterior torso', 
                             'upper extremity']]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index: int)->Tuple[torch.Tensor, Any]:
        image_path = os.path.join(self.image_dir, self.df['isic_id'].iloc[index] +'.jpg')
        image= Image.open(image_path).convert('RGB')
        image_tensor= self.transform(image)

        tabular_cont= torch.tensor(self.df_cont.iloc[index].values,  dtype=torch.float)
        tabular_bin= torch.tensor(self.df_bin.iloc[index].values,  dtype=torch.long)

        label= self.df['target'][index]

        return image_tensor, tabular_cont, tabular_bin, label

def partial_auc(labels, predictions, min_tpr: float=0.80):
    labels = numpy.asarray(labels)
    predictions = numpy.asarray(predictions)

    labels = numpy.abs(labels - 1)
    predictions = 1.0 - predictions

    max_fpr = 1-min_tpr
    partial_auc_scaled = roc_auc_score(labels, predictions, max_fpr=max_fpr)
    return 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

def train_evaluate_model(data:pandas.DataFrame, config: Dict[str, str], splits: List[Any], model: Union[ResUNet, ResUNetWithTabular], **kwargs):
    assert isinstance(model, (ResUNet, ResUNetWithTabular)), 'Model Not Supported'
    
    model.to('cuda')
    model_name= config['model']['name']
    train_steps= config['training']['train_steps']
    valid_steps= config['training']['valid_steps']
    epochs= config['training']['epochs']
    patience= config['training']['patience']
    log_dir= os.path.join(os.path.dirname(__file__), config['logging']['log_dir'])
    img_dir= os.path.join(os.path.dirname(__file__), config['data']['image_dir'])

    optimizer= AdamW(model.parameters(), lr= config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                  factor=0.1, patience=patience)
    criterion= torch.nn.BCEWithLogitsLoss()
    scaler= GradScaler()

    num_runs=len(fnmatch.filter(os.listdir(os.path.join(log_dir, 'runs')), f"{model_name}_*"))
    num_cks= len(fnmatch.filter(os.listdir(os.path.join(log_dir, 'checkpoints')), f"{model_name}_*"))
    checkpoint_name= f'{model_name}_ck{num_cks}'

    os.mkdir(os.path.join(log_dir, f'runs/{model_name}_run{num_runs}/'))
    os.mkdir(os.path.join(log_dir, f'checkpoints/{model_name}_ck{num_runs}/'))
    writer= SummaryWriter(log_dir=os.path.join(os.path.join(log_dir, f'runs/{model_name}_run{num_runs}/')))


    for fold, (train_idx, valid_idx) in enumerate(tqdm(splits, desc="Splits")):
        print(f"Fold {fold + 1}")

        train_data= data.iloc[train_idx].reset_index(drop=True)
        valid_data= data.iloc[valid_idx].reset_index(drop=True)

        train_dataset= SkinDataset(train_data, TRAIN_TRANSFORM, img_dir)
        valid_dataset= SkinDataset(valid_data, TRANSFORM, img_dir)

        train_loader= DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)
        valid_loader= DataLoader(valid_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)

        best_val_loss = float('inf')
        for epoch in tqdm(range(epochs), desc='Epochs'):

            model.train()
            train_loss= 0.0
            with  tqdm(total= train_steps, desc='Training Batches', leave=False) as bar:
                for step, (images, tabular_cont, tabular_bin, labels) in enumerate(train_loader):
                    if step >= train_steps:
                        break

                    if isinstance(model, ResUNet):
                        images, labels= images.to('cuda'), labels.to('cuda').float()

                        with autocast():
                            outputs= model(images)
                            loss= criterion(outputs.squeeze(dim= 1), labels)

                    elif isinstance(model, ResUNetWithTabular):
                        tabular_cont, tabular_bin= tabular_cont.to('cuda'), tabular_bin.to('cuda')
                        images, labels= images.to('cuda'), labels.to('cuda').float()

                        with autocast():
                            outputs= model(images, tabular_cont, tabular_bin)
                            loss= criterion(outputs.squeeze(dim= 1), labels)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm(model.parameters() , 
                        max_norm=config['training']['max_norm'], norm_type=config['training']['norm_type'])
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss+= loss.item()

                    writer.add_scalar('Loss/train', loss.item(), (fold * epochs * train_steps + epoch * train_steps + step))
                    bar.set_postfix(loss= loss.item(), refresh=True)
                    bar.update(1)

            train_loss= train_loss / train_steps
            print(f' Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4e}')
    
            model.eval()
            val_loss=0.0
            all_labels, all_preds = [], []
            with torch.no_grad():
                with tqdm(total= valid_steps, desc='Validation Batches', leave=False) as bar:
                    for step, (images, tabular_cont, tabular_bin, labels) in enumerate(valid_loader):
                        if step >= valid_steps:
                            break
                        
                        if isinstance(model, ResUNet):
                            images, labels= images.to('cuda'), labels.to('cuda').float()

                            with autocast():
                                outputs= model(images)
                                loss= criterion(outputs.squeeze(dim= 1), labels)

                        elif isinstance(model, ResUNetWithTabular):
                            tabular_cont, tabular_bin= tabular_cont.to('cuda'), tabular_bin.to('cuda')
                            images, labels= images.to('cuda'), labels.to('cuda').float()

                            with autocast():
                                outputs= model(images, tabular_cont, tabular_bin)
                                loss= criterion(outputs.squeeze(dim= 1), labels)

                        probabilities = torch.sigmoid(outputs)

                        val_loss+= loss.item()

                        all_labels.append(labels.detach().cpu().numpy())
                        all_preds.append(probabilities.detach().cpu().numpy())

                        writer.add_scalar('Loss/valid', loss.item(), fold * epochs * valid_steps + epoch * valid_steps+ step)
                        bar.set_postfix(loss= loss.item(), refresh=True)
                        bar.update(1)

                val_loss = val_loss / valid_steps

                all_labels = numpy.concatenate(all_labels)
                all_preds = numpy.concatenate(all_preds)
                unique_labels = numpy.unique(all_labels)
                
                if len(unique_labels) < 2:
                    print(f' Warning: Only one class present in labels. Unique labels: {unique_labels}')
                else: p_auc= partial_auc(all_labels, all_preds)
                
                writer.add_scalar('Partial AUC/valid', p_auc, fold*epochs + epoch)

            scheduler.step(val_loss)
            torch.cuda.empty_cache()
            gc.collect()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0

                torch.save(model.state_dict(), os.path.join(log_dir, f'checkpoints/{checkpoint_name}/{model_name}_fold_{fold+1}.pt'))
                print(f' Saved best model for fold {fold + 1} with validation loss: {val_loss:.4e}')

            else:
                patience += 1
                if patience >= 10:
                    print(f' Early stopping triggered after {fold*epochs + epoch + 1} total epochs')
                    break

    torch.save(model.state_dict(), os.path.join(log_dir, f'checkpoints/{checkpoint_name}/{model_name}_final.pt'))
    return checkpoint_name, num_runs

def test_model(data:pandas.DataFrame, config: Dict[str, str],  model: Union[ResUNet, ResUNetWithTabular], **kwargs)-> Any:
    img_dir= os.path.join(os.path.dirname(__file__), config['data']['image_dir'])
    log_dir= os.path.join(os.path.dirname(__file__), config['logging']['log_dir'])

    test_dataset= SkinDataset(data, TRANSFORM, img_dir)
    test_loader= DataLoader(test_dataset, batch_size=config['testing']['batch_size'], shuffle=False, num_workers=config['testing']['num_workers'])
    all_labels, all_preds= [], []

    ck_fp= os.path.join(log_dir, f"checkpoints/{config['testing']['ck']}/{config['model']['name']}_final.pt")
    run_fp_for_metric= os.path.join(log_dir, f"runs/{config['model']['name']}_run{config['testing']['run']}/metric.json")

    state_dict= torch.load(ck_fp)
    model.load_state_dict(state_dict)
    model.to('cuda').eval()

    for images, tabular_cont, tabular_bin, labels in tqdm(test_loader,  desc='Test Batches', leave=False):
        if isinstance(model, ResUNet):
            images, labels= images.to('cuda'), labels.to('cuda').float()
            outputs= model(images)
                        
        elif isinstance(model, ResUNetWithTabular):
            tabular_cont, tabular_bin= tabular_cont.to('cuda'), tabular_bin.to('cuda')
            images, labels= images.to('cuda'), labels.to('cuda').float()
            outputs= model(images, tabular_cont, tabular_bin)

        probabilities= torch.sigmoid(outputs).detach().cpu()
        all_labels.append(labels.detach().cpu())
        all_preds.append(probabilities)

    all_labels= numpy.concatenate(all_labels)
    all_preds= numpy.concatenate(all_preds)
    unique_labels = numpy.unique(all_labels)

    if len(unique_labels) < 2:
        print(f' Warning: Only one class present in labels. Unique labels: {unique_labels}')
    else:
        pAUC= partial_auc(all_labels, all_preds)
        with open(run_fp_for_metric, 'w') as file:
            json.dump({'Partial AUC': round(pAUC, 3)}, file)
        return pAUC
    
    return None

def train_evaluate_test(train_evaluate_args: Dict[Any, Any], test_args: Dict[Any, Any]):
    ck, run= wrap_cuda_streamer(train_evaluate_model, train_evaluate_args)
    if test_args is not None:
        test_args['config']['testing'].update({'ck': ck, 'run': run})
        return wrap_cuda_streamer(test_model, test_args)

def wrap_cuda_streamer(fn: Callable[[Any], Any], kwargs: Dict[str, Any]):
    stream= torch.cuda.Stream()

    with torch.cuda.stream(stream):
        res= fn(**kwargs)

    torch.cuda.current_stream().wait_stream(stream)
    del stream
    torch.cuda.empty_cache()
    return res

def load_config(fp):
    with open(fp, 'r') as file:
        return yaml.safe_load(file)
    
def set_seed(seed: int= 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def oversample(data: pandas.DataFrame, seed: int):
    ros= RandomOverSampler(random_state=seed)
    X= data.drop(columns=['target'])
    y= data['target']

    X, y= ros.fit_resample(X, y)

    return pandas.concat([X, pandas.Series(y, name='target')], axis=1)

def prepare(data: pandas.DataFrame, seed:int):
    X_train, X_test, y_train, y_test= train_test_split(
        data.drop(columns=['target']),
        data['target'], 
        test_size=1059,
        random_state=seed, 
        stratify= data['target']
    )
    
    train_df= pandas.concat([X_train, y_train], axis=1).reset_index(drop=True)
    train_df= oversample(train_df, seed)
    test_df= pandas.concat([X_test, y_test], axis=1).reset_index(drop=True)
    return train_df, test_df