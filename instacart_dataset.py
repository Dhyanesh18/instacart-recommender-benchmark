from torch.utils.data import Dataset
import numpy as np

class InstacartDataset(Dataset):
    """
    PyTorch Dataset for Instacart user–product interactions.

    - Maps raw user_id and product_id to continuous indices (0..N-1).
    - These indices can be used with nn.Embedding layers instead of one-hot vectors.
    - Returns (user_idx, product_idx, label) for each sample.
    """
    def __init__(self, df):
        self.df = df
        self.user_map = {original_id:i for i, original_id in enumerate(df['user_id'].unique())}
        self.product_map = {original_id:i for i, original_id in enumerate(df['product_id'].unique())}
        self.df['user_idx'] = df['user_id'].map(self.user_map)
        self.df['product_idx'] = df['product_id'].map(self.product_map)
        self.users = self.df['user_idx'].values
        self.products = self.df['product_idx'].values
        self.labels = self.df['y'].values.astype(np.float32)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'user' : self.users[idx],
            'product': self.products[idx],
            'label': self.labels[idx]
        }