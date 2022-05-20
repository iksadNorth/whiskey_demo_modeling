# %%
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from recbole.quick_start.quick_start import load_data_and_model
from recbole.data.dataloader.user_dataloader import UserDataLoader

import pandas as pd
import numpy as np
import torch
from pathlib import Path

import yaml
with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)

# %%
class Collector():
    def __init__(self, goods:list=[], poors:list=[], dataloader:UserDataLoader=None) -> None:
        self.goods = goods
        self.poors = poors
        self.config, self.model, self.dataset, self.dataloader, _, _ = load_data_and_model(
            model_file=Path(CONFIG['dir_model_saved']) / CONFIG['file_model_used'],
        )
        self.config_vae, self.model_vae, _, _, _, _ = load_data_and_model(
            model_file=Path(CONFIG['dir_model_saved']) / CONFIG['file_model_vae_used'],
        )
        if dataloader:
            self.dataloader = dataloader
    
    def _popularity(self, k:int=10) -> list:
        list_pop = []
        iterator = pd.read_csv(CONFIG['dir_Pop']).iterrows()
        
        for idx, row in iterator:
            whiskey = row['whiskey']
            if len(list_pop) == k:
                break
            if whiskey in self.goods:
                continue
            elif whiskey in self.poors:
                continue
            list_pop.append(whiskey)
        return list_pop
    
    def _recbole(self, user:str, k:int=10) -> list:
        uid_series = self._encode_user([user])
        
        topk_score, topk_iid_list = full_sort_topk(
            uid_series, self.model, 
            self.dataloader, k=k, 
            device=self.config['device'])
        
        return self._decode(topk_iid_list.cpu())[0]
    
    def _recvae_full_sort_predict(self) -> torch.Tensor:
        uid_series_good = self._encode(self.goods)
        uid_series_poor = self._encode(self.poors)
        
        rating_matrix = self._make_rating_matrix(uid_series_good)
        scores, _, _, _ = self.model_vae.forward(rating_matrix, self.model_vae.dropout_prob)

        # [PAD], self.goods, self.poors에 해당하는 모든 아이템 제외
        scores[:, 0] = -np.inf
        scores[:, uid_series_good] = -np.inf
        scores[:, uid_series_poor] = -np.inf
        
        return scores
    
    def _recvae_topk(self, k:int=10) -> tuple:
        scores = self._recvae_full_sort_predict()
        topk_scores, topk_index = torch.topk(scores, k)
        return self._decode(topk_index.cpu()).tolist()[0]
    
    def _make_rating_matrix(self, uid_series:np.ndarray) -> torch.Tensor:
        n_items = self.dataset.num(self.dataset.iid_field)
        device=self.config_vae['device']
        
        col_indices = torch.Tensor(uid_series).to(device).long()
        row_indices = torch.zeros_like(col_indices).to(device).long()
        cell_values = torch.tensor(1.).to(device)
        
        rating_matrix = torch.zeros(1).to(device).repeat(1, n_items)
        rating_matrix.index_put_((row_indices, col_indices), cell_values)
        
        return rating_matrix
    
    def topk(self, k:int=10) -> list:
        return self._popularity(k)
    
    def retrain(self) -> None:
        raise NotImplementedError
    
    def _encode(self, item:list) -> np.ndarray:
        return self.dataset.token2id(self.dataset.iid_field, item)
    
    def _encode_user(self, user:list) -> np.ndarray:
        return self.dataset.token2id(self.dataset.uid_field, user)
    
    def _decode(self, array_id:list) -> np.ndarray:
        return self.dataset.id2token(self.dataset.iid_field, array_id)


# %%
if __name__ == '__main__':
    agent = Collector()
    list_pop = agent.topk(10)
    print(list_pop)

# %%
