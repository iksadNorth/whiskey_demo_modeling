# %%
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# %%
# config
dir_src = Path('/opt/ml/workspace/src')
scaler = MinMaxScaler()

# %%
# load source
df_images = pd.read_csv(dir_src / 'whiskey_images_links.csv', sep='$')
df_whisky_w_tags = pd.read_csv(dir_src / 'whisky_w_tags.csv', index_col='Unnamed: 0')
df_interaction = pd.read_csv(dir_src / 'user_whiskey_interaction.csv', sep='$')
df_integration = pd.DataFrame([])

# MinMaxScaler 방식으로 정규화 진행.
cols_taste = ['body','sweet','sherry','malt','aperitif','smoky','pungent','fruity','honey','floral','spicy','medicinal','nutty','winey']
df_whisky_w_tags_taste = df_whisky_w_tags[cols_taste]
df_whisky_w_tags[cols_taste] = (df_whisky_w_tags_taste - df_whisky_w_tags_taste.min(axis=0)) / (df_whisky_w_tags_taste.max(axis=0) - df_whisky_w_tags_taste.min(axis=0))

# %%
# preprocessing
df_interaction = df_interaction[['whiskey', 'user', 'rating']]
df_integration = df_whisky_w_tags.merge(df_images, left_on='Whisky', right_on='whiskey', how='left')

# %%
# 결측치 확인 cell
df_integration[df_integration['images'].isna()]
# => image 열에서 50개의 데이터가 Nan값을 가짐.

# %%
# 전처리 후 저장.
df_interaction.to_csv('/opt/ml/workspace/src/src_processed/interaction.csv', index=False, sep='$')
df_integration.to_csv('/opt/ml/workspace/src/src_processed/integration.csv', index=False, sep='$')

# %%
