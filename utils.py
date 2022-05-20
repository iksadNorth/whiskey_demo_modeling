# %%
import pandas as pd

import yaml
with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)

# %%
class Collector():
    def __init__(self, goods:list=[], poors:list=[]) -> None:
        self.goods = goods
        self.poors = poors
    
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
    
    def _recbole(self, k:int=10) -> list:
        raise NotImplementedError
    
    def topk(self, k:int=10) -> list:
        return self._popularity(k)
    
    def retrain(self) -> None:
        raise NotImplementedError


# %%
if __name__ == '__main__':
    agent = Collector()
    list_pop = agent.topk(10)
    print(list_pop)

# %%
