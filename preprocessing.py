# %%
import pandas as pd

import yaml
with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)


# %%
columns         = CONFIG['columns']
columns_types   = CONFIG['columns_types']
sep = CONFIG['sep']

dir_Top_1000_Whiskey_Data = CONFIG['dir_Top_1000_Whiskey_Data']
dir_whiskey_dataset = CONFIG['dir_whiskey_dataset']
dir_inter = CONFIG['dir_inter']

# %%
# .inter 생성을 위한 cell
df_Top_1000_Whiskey_Data = pd.read_csv(dir_Top_1000_Whiskey_Data, index_col='Unnamed: 0')
df_inter = df_Top_1000_Whiskey_Data[columns]

columns_with_types = [f"{col}:{_type}" for col, _type in zip(columns, columns_types)]
df_inter.columns = columns_with_types

# %%
# .inter 생성을 위한 cell
df_inter.to_csv(dir_inter, index=False, sep=sep)

# %%
# 데이터 인기도순 추천을 위한 전처리 과정
df_pop = df_Top_1000_Whiskey_Data[['user', 'whiskey']]\
    .groupby('whiskey').count()\
        .sort_values('user', ascending=False)\
            .reset_index()
            
df_pop.to_csv(CONFIG['dir_Pop'], index=False)
# %%
