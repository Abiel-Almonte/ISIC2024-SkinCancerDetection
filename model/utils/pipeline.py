import torch
import pandas
import numpy
import os
import gc
import json
import fnmatch
from typing import Tuple, Any, Union, Dict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW, Adam, SGD, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from architectures import ISICModel
from .criterion import get_lossfn, partial_auc
from .data import SkinDataset, TRAIN_TRANSFORM, TRANSFORM

__all__= ['train_evaluate_model', 'test_model']

def get_optim(name: str)-> torch.optim.Optimizer:
    if name == 'Adam':
        return Adam
    elif name == 'AdamW':
        return AdamW
    elif name =='SGD':
        return SGD
    elif name== 'RMS':
        return RMSprop
    else:
        raise NotImplemented('Optimizer Not Supported')
    
def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable

def gradual_unfreeze(model, num_layers_to_unfreeze):
    set_trainable(model, False)
    
    children = list(model.children())
    for child in children[-num_layers_to_unfreeze:]:
        set_trainable(child, True)

def train_evaluate_model(
    train_data: pandas.DataFrame, 
    valid_data: pandas.DataFrame, 
    config: Dict[str, Any], 
    model: Union[ISICModel], 
    **kwargs
) -> int:
    """
    Trains and evaluates the model using the provided training and validation datasets.

    Args:
        train_data (pandas.DataFrame): DataFrame containing the training data.
        valid_data (pandas.DataFrame): DataFrame containing the validation data.
        config (Dict[str, Any]): Configuration dictionary containing model and training parameters.
        model (Union[ModelProtocol]): The model to be trained and evaluated.
        **kwargs: Additional keyword arguments for model training.
    """
    assert isinstance(model, ISICModel), 'Model Not Supported'

    model.to('cuda')

    model_name= config['model']['name']
    train_steps= config['training']['train_steps']
    valid_steps= config['training']['valid_steps']
    epochs= config['training']['epochs']
    patience= config['training']['patience']


    log_dir= os.path.join(os.path.dirname(__file__), config['logging']['log_dir'])
    img_dir= os.path.join(os.path.dirname(__file__), config['data']['image_dir'])


    optimizer= get_optim(config['training']['optim']['name'])(model.parameters(), **config['training']['optim']['parameters'])
    scheduler= ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)
    #scheduler= CosineAnnealingLR(optimizer, T_max=10)
    criterion= get_lossfn(config['training']['loss_fn'])()
    tuple_criterion= get_lossfn('cont')()
    scaler= GradScaler()

    num_runs= len(fnmatch.filter(os.listdir(os.path.join(log_dir, 'runs')), f"{model_name}_*"))
    num_cks= len(fnmatch.filter(os.listdir(os.path.join(log_dir, 'checkpoints')), f"{model_name}_*"))
    
    assert num_runs== num_cks, 'Log Discrepancy. Ensure there is a checkpoint for every tensorboard log.'

    checkpoint_name= f'{model_name}_ck{num_cks}'
    run_name= f'{model_name}_run{num_runs}'

    os.makedirs(os.path.join(log_dir, f'runs/{run_name}/'), exist_ok= True)
    os.makedirs(os.path.join(log_dir, f'checkpoints/{checkpoint_name}/'), exist_ok= True)
    
    writer= SummaryWriter((os.path.join(log_dir, 'runs', run_name)))

    train_dataset= SkinDataset(train_data, TRAIN_TRANSFORM, img_dir)
    valid_dataset= SkinDataset(valid_data, TRANSFORM, img_dir)

    class_counts= numpy.bincount(train_dataset.df['target'])
    class_weights= 1. / class_counts
    train_sample_weights= class_weights[train_dataset.df['target']]
    train_sampler= WeightedRandomSampler(train_sample_weights, len(train_sample_weights))
    
    train_loader= DataLoader(
        train_dataset, 
        batch_size= config['training']['batch_size'], 
        sampler=train_sampler, 
        num_workers= config['training']['num_workers'], 
        pin_memory= True
    )
    
    valid_loader= DataLoader(
        valid_dataset, 
        batch_size= config['training']['batch_size'], 
        shuffle= True, 
        num_workers= config['training']['num_workers'], 
        pin_memory= True
    )

    total_layers = len(list(model.children()))
    layers_to_unfreeze= 0
    best_valid= float('inf')
    paucs= []
    for epoch in tqdm(range(epochs), desc= 'Epochs'):
        model.train()
        train_loss = 0.0
        all_labels, all_preds = [], []
        
        with tqdm(total= train_steps, desc= 'Training Batches', leave= False) as bar:
            for step, (images, tabular_cont, tabular_bin, labels) in enumerate(train_loader):
                if step>= train_steps:
                    break

                tabular_cont, tabular_bin= tabular_cont.to('cuda'), tabular_bin.to('cuda')
                images, labels= images.to('cuda'), labels.to('cuda').float()

                with autocast():
                    outputs= model(images, tabular_cont, tabular_bin)
                    if isinstance(outputs, Tuple):
                        outputs, proj = outputs
                        loss= tuple_criterion(proj, labels) + criterion(outputs.squeeze(dim= 1), labels)
                    else:
                        loss= criterion(outputs.squeeze(dim=1), labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
                scaler.step(optimizer)
                scaler.update()

                train_loss+= loss.item()
                writer.add_scalar('Loss/train', loss.item(), (epoch* train_steps+ step))

                probabilities= torch.sigmoid(outputs.squeeze(dim=1))
                all_labels.append(labels.detach().cpu().numpy())
                all_preds.append(probabilities.detach().cpu().numpy())

                bar.set_postfix(loss= loss.item(), refresh= True)
                bar.update(1)

        #if (epoch+1) % 4== 0:
        #    layers_to_unfreeze= min(layers_to_unfreeze + 2, total_layers)
        #    gradual_unfreeze(model, layers_to_unfreeze)
        #    print(f' Layers unfrozen: {layers_to_unfreeze}')

        train_loss/= train_steps
        all_labels= numpy.concatenate(all_labels)
        all_preds= numpy.concatenate(all_preds)
        unique_labels= numpy.unique(all_labels)
        
        if len(unique_labels)< 2:
            print(f' Warning: Only one class present in labels. Unique labels: {unique_labels}')
        else:
            p_auc= partial_auc(all_labels, all_preds).item()
            paucs.append(p_auc)
            writer.add_scalar('Partial AUC/train', p_auc, epoch* train_steps+ step)
        
        print(f' Epoch {epoch+ 1}/{epochs}, Loss: {train_loss:.4e}')

        model.eval()
        val_loss= 0.0
        all_labels, all_preds= [], []
        
        with torch.inference_mode():
            with tqdm(total= valid_steps, desc= 'Validation Batches', leave= False) as bar:
                for step, (images, tabular_cont, tabular_bin, labels) in enumerate(valid_loader):
                    if step>= valid_steps:
                        break

                    tabular_cont, tabular_bin= tabular_cont.to('cuda'), tabular_bin.to('cuda')
                    images, labels= images.to('cuda'), labels.to('cuda').float()

                    with autocast():
                        outputs= model(images, tabular_cont, tabular_bin)
                        if isinstance(outputs, Tuple):
                            outputs, proj = outputs
                            loss= tuple_criterion(proj, labels) + criterion(outputs.squeeze(dim= 1), labels)
                        else:
                            loss= criterion(outputs.squeeze(dim=1), labels)

                    probabilities= torch.sigmoid(outputs.squeeze(dim= 1))
                    val_loss+= loss.item()

                    all_labels.append(labels.detach().cpu().numpy())
                    all_preds.append(probabilities.detach().cpu().numpy())

                    writer.add_scalar('Loss/valid', loss.item(), epoch* valid_steps+ step)
                    bar.set_postfix(loss= loss.item(), refresh= True)
                    bar.update(1)

            val_loss/= valid_steps
            all_labels= numpy.concatenate(all_labels)
            all_preds= numpy.concatenate(all_preds)
            unique_labels= numpy.unique(all_labels)
            
            if len(unique_labels)< 2:
                print(f' Warning: Only one class present in labels. Unique labels: {unique_labels}')
            else:
                p_auc= partial_auc(all_labels, all_preds).item()
                paucs.append(p_auc)
                writer.add_scalar('Partial AUC/valid', p_auc, epoch)
            
            scheduler.step(val_loss)
            
            torch.cuda.empty_cache()
            gc.collect()

            if best_valid> val_loss:
                best_valid= val_loss
                patience= 0

                model_dict={
                    'model_state_dict': model.state_dict(),
                    'avg_pauc': round(numpy.mean(paucs).item(), 3),
                    'loss': val_loss,
                    'config': config,
                    'model_architecture': model
                }

                torch.save(model_dict, os.path.join(log_dir, f'checkpoints/{checkpoint_name}/{model_name}.pth'))
                print(f' Saved best model with validation loss: {val_loss:.4e}')
            else: pass
                #patience+= 1
                #if patience>= 10:
                #    print(f' Early stopping triggered after {epoch+ 1} total epochs')
                #    break

    model_dict={
        'model_state_dict': model.state_dict(),
        'avg_pauc': round(numpy.mean(paucs).item(), 3),
        'loss': val_loss,
        'config': config,
        'model_architecture': model
    }

    torch.save(model_dict, os.path.join(log_dir, f'checkpoints/{checkpoint_name}/{model_name}_final.pth'))

    return num_runs

def test_model(
    data: pandas.DataFrame, 
    config: Dict[str, str],  
    model: Union[ISICModel], 
    **kwargs
) -> Any:
    """
    Tests the model on the provided dataset and saves the results.

    Args:
        data (pandas.DataFrame): DataFrame containing the test data.
        config (Dict[str, str]): Configuration dictionary containing model and testing parameters.
        model (Union[ModelProtocol]): The model to be tested.
        **kwargs: Additional keyword arguments for model testing.

    Returns:
        Any: Partial AUC score of the model on the test data.
    """
    img_dir= os.path.join(os.path.dirname(__file__), config['data']['image_dir'])
    log_dir= os.path.join(os.path.dirname(__file__), config['logging']['log_dir'])

    unique_labels= numpy.unique(data['target'].values)
    if len(unique_labels) < 2:
        print(f' Warning: Only one class present in labels. Unique labels: {unique_labels}')
        return None

    test_dataset= SkinDataset(data, TRANSFORM, img_dir)
    test_loader= DataLoader(
        test_dataset, 
        batch_size= config['testing']['batch_size'], 
        shuffle= False, 
        num_workers= config['testing']['num_workers']
    )
    
    all_labels, all_preds= [], []

    ck_fp= os.path.join(log_dir, f"checkpoints/{config['model']['name']}_ck{config['testing']['run']}/{config['model']['name']}_final.pth")
    run_fp_for_metric= os.path.join(log_dir, f"runs/{config['model']['name']}_run{config['testing']['run']}/metric.json")

    checkpoint= torch.load(ck_fp)
    model= checkpoint['model_architecture']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda')
    model.eval()

    assert isinstance(model, ISICModel), 'Model Not Supported'

    with torch.inference_mode():
        for images, tabular_cont, tabular_bin, labels in tqdm(test_loader, desc='Test Batches', leave=False):
            tabular_cont, tabular_bin= tabular_cont.to('cuda'), tabular_bin.to('cuda')
            images, labels= images.to('cuda'), labels.to('cuda').float()
            outputs= model(images, tabular_cont, tabular_bin)
            
            if isinstance(outputs, Tuple):
                outputs, _, _= outputs

            probabilities= torch.sigmoid(outputs.squeeze(1))
            all_labels.append(labels.detach().cpu().numpy())
            all_preds.append(probabilities.detach().cpu().numpy())

    all_labels= numpy.concatenate(all_labels)
    all_preds= numpy.concatenate(all_preds)
    unique_labels= numpy.unique(all_labels)

    pAUC= partial_auc(all_labels, all_preds)
    
    with open(run_fp_for_metric, 'w') as file:
        json.dump({'Partial AUC': round(pAUC, 3), 'Testing Samples': len(data)}, file)

    return pAUC