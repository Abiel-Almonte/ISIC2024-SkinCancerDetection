import os
import pandas
import torch
from typing import Tuple, Any
from PIL import Image
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from torchvision import transforms
from torch.utils.data import Dataset

__all__ = ['TRAIN_TRANSFORM', 'TRANSFORM', 'SkinDataset', 'oversample', 'prepare']

class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(336),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5),
    GaussianNoise(0., 0.1),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TRANSFORM = transforms.Compose([
    transforms.Resize(336),
    # transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SkinDataset(Dataset):
    def __init__(self, df: pandas.DataFrame, transform: transforms.Compose, image_dir: str) -> None:
        """
        Initializes the SkinDataset.

        Parameters:
            df (pandas.DataFrame): DataFrame containing image paths and labels.
            transform (transforms.Compose): Transformations to apply to images.
            image_dir (str): Directory containing image files.
        """
        super().__init__()
        self.df = df
        self.transform = transform
        self.image_dir = os.path.join(os.path.dirname(__file__), image_dir)
        
        self.df_cont = self.df.drop(columns=[
            'isic_id', 'target', 'patient_id', 'sex', 'anterior torso',
            'head/neck', 'lower extremity', 'posterior torso', 'upper extremity'
        ])
        self.df_bin = self.df[['sex', 'anterior torso', 'head/neck', 'lower extremity', 
                               'posterior torso', 'upper extremity']]
        self.df_cont= (self.df_cont - self.df_cont.mean())/ self.df_cont.std()

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        image_path = os.path.join(self.image_dir, self.df['isic_id'].iloc[index] + '.jpg')
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        tabular_cont = torch.tensor(self.df_cont.iloc[index].values, dtype=torch.float)
        tabular_bin = torch.tensor(self.df_bin.iloc[index].values, dtype=torch.long)

        label = self.df['target'].iloc[index]

        return image_tensor, tabular_cont, tabular_bin, label

def oversample(data: pandas.DataFrame, seed: int) -> pandas.DataFrame:
    """
    Performs random oversampling to balance the dataset.

    Parameters:
        data (pandas.DataFrame): The dataset to oversample.
        seed (int): Random seed for reproducibility.

    Returns:
        pandas.DataFrame: The oversampled dataset.
    """
    ros = RandomOverSampler(random_state=seed)
    X = data.drop(columns=['target'])
    y = data['target']

    X_resampled, y_resampled = ros.fit_resample(X, y)

    return pandas.concat([X_resampled, pandas.Series(y_resampled, name='target')], axis=1)

def undersample(df, seed:int, N=117900):
    """
    Performs random undersampling to balance the dataset.

    Parameters:
        data (pandas.DataFrame): The dataset to oversample.
        seed (int): Random seed for reproducibility.

    Returns:
        pandas.DataFrame: The undersampled dataset.
    """
    p_cases = df[df['target'] == 1]
    n_cases = df[df['target'] == 0]

    n_cases = n_cases.sample(n=N, random_state=seed)

    df = pandas.concat([n_cases, p_cases])

    return df

def prepare(data: pandas.DataFrame, test_size: int | float , seed: int, use_oversample: bool = False, use_undersample:bool= True) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Prepares the dataset by splitting it into training and testing sets, and optionally oversampling the training set.

    Parameters:
        data (pandas.DataFrame): The dataset to prepare.
        test_size (int): The proportion of the dataset to use for testing.
        seed (int): Random seed for reproducibility.
        use_oversample (bool): Whether to oversample the training set.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: The training and testing datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=['target']),
        data['target'],
        test_size=test_size,
        random_state=seed,
        stratify=data['target']
    )
    
    train_df = pandas.concat([X_train, y_train], axis=1).reset_index(drop=True)
    if use_oversample:
        train_df= oversample(train_df, seed)
    if use_undersample:
        train_df= undersample(train_df, seed) 
    test_df = pandas.concat([X_test, y_test], axis=1).reset_index(drop=True)
    
    return train_df, test_df
