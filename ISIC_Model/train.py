import torch.utils
import torch
from arch import ResUNet, ResUNetWithTabular
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
from typing import Tuple, Any, Union, List, Dict
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
import pandas, numpy, fnmatch, os, gc

IMAGE_DIR= os.path.join(os.path.dirname(__file__), 
                        '../data/train-image/image')
LOG_DIR= os.path.join(os.path.dirname(__file__), 'logs')

train_transforms = transforms.Compose([
    transforms.Resize(480),
    transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(480),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SkinDataset(Dataset):
    def __init__(self, df:pandas.Series, transform, image_dir:str) -> None:
        super().__init__()
        self.df= df
        self.image_dir= image_dir
        self.transform= transform
        
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

def oversample(data: pandas.DataFrame):
    ros= RandomOverSampler(random_state=42)
    X= data.drop(columns=['target'])
    y= data['target']

    X, y= ros.fit_resample(X, y)

    return pandas.concat([X, pandas.Series(y, name='target')], axis=1)

def partial_auc(labels, predictions, min_tpr: float=0.80):
    labels = numpy.asarray(labels)
    predictions = numpy.asarray(predictions)

    labels = numpy.abs(labels - 1)
    predictions = 1.0 - predictions

    max_fpr = 1-min_tpr

    partial_auc_scaled = roc_auc_score(labels, predictions, max_fpr=max_fpr)
    
    return 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

def train_evaluate_model(data:pandas.DataFrame, epochs:int, config: Dict[str, str], splits: List[Any], model: Union[ResUNet, ResUNetWithTabular]):
    assert isinstance(model, (ResUNet, ResUNetWithTabular)), 'Model Not Supported'
    
    model.to('cuda')
    model_name= config['model']['name']
    train_steps= config['training']['train_steps']
    valid_steps= config['training']['valid_steps']
    
    optimizer= AdamW(model.parameters(), lr= config['training']['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config['training']['patience'])
    criterion= torch.nn.BCEWithLogitsLoss()
    scaler= GradScaler()

    num_runs=len(fnmatch.filter(os.listdir(os.path.join(LOG_DIR, 'runs')), f"{model_name}*"))
    num_cks= len(fnmatch.filter(os.listdir(os.path.join(LOG_DIR, 'checkpoints')), f"{model_name}*"))
    checkpoint_name= f'{model_name}_ck{num_cks}'

    if not os.path.isdir(os.path.join(LOG_DIR, f'runs/{model_name}_run{num_runs}/')):
        os.mkdir(os.path.join(LOG_DIR, f'runs/{model_name}_run{num_runs}/'))
    if not os.path.isdir(os.path.join(LOG_DIR, f'checkpoints/{model_name}_ck{num_runs}/')):
        os.mkdir(os.path.join(LOG_DIR, f'checkpoints/{model_name}_ck{num_runs}/'))

    writer= SummaryWriter(log_dir=os.path.join(os.path.join(LOG_DIR, f'runs/{model_name}_run{num_runs}/')))

    for fold, (train_idx, valid_idx) in enumerate(tqdm(splits, desc="Splits")):
        print(f"Fold {fold + 1}")

        train_data= data.iloc[train_idx].reset_index(drop=True)
        valid_data= data.iloc[valid_idx].reset_index(drop=True)

        train_dataset= SkinDataset(train_data, train_transforms, config['data']['image_dir'])
        valid_dataset= SkinDataset(valid_data, valid_transforms, config['data']['image_dir'])

        train_loader= DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=20, pin_memory=True)
        valid_loader= DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=20, pin_memory=True)

        best_val_loss = float('inf')
        patience= 0
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
                    torch.nn.utils.clip_grad_norm(model.parameters() , max_norm=1, norm_type=1)
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss+= loss.item()

                    writer.add_scalar('Loss/train', loss.item(), fold*epochs + epoch*train_steps + step)
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

                        all_labels.append(labels.cpu().numpy())
                        all_preds.append(probabilities.cpu().numpy())

                        writer.add_scalar('Loss/valid', loss.item(), fold*epochs + epoch*valid_steps + step)
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

                torch.save(model.state_dict(), os.path.join(LOG_DIR, f'checkpoints/{checkpoint_name}/best_{model_name}_fold_{fold+1}.pt'))
                print(f' Saved best model for fold {fold + 1} with validation loss: {val_loss:.4e}')

            else:
                patience += 1
                if patience >= 10:
                    print(f' Early stopping triggered after {fold*epochs + epoch + 1} total epochs')
                    break

    torch.save(model.state_dict(), os.path.join(LOG_DIR, f'checkpoints/{checkpoint_name}/final_{model_name}.pt'))

if __name__ =='__main__':
    from sklearn.model_selection import StratifiedGroupKFold
    from train import train_evaluate_model
    import warnings, pandas, yaml

    warnings.filterwarnings('ignore')

    def load_config(fp):
        with open(fp, 'r') as file:
            return yaml.safe_load(file)
    
    config= load_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    metadata= pandas.read_csv(os.path.join(os.path.dirname(__file__), config['data']['metadata_file']))
    metadata= oversample(metadata)

    splits= list(StratifiedGroupKFold(n_splits=2).split(metadata, metadata['target'], groups= metadata['patient_id']))

    if config['model']['name'] == 'ResUNetWithTabular':
        model = ResUNetWithTabular(config['model']['cont_features'], config['model']['bin_features'])
    elif config['model']['name'] == 'ResUNet':
        model = ResUNet()
    else:
        raise ValueError(f"Unsupported model: {config['model']['name']}")
    
    train_evaluate_model(metadata, config, splits, model)