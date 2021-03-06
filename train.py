# %%
from recbole.quick_start import run_recbole

from pathlib import Path

import yaml
with open('config.yaml', 'r') as f:
    CONFIG = yaml.load(f)

# %%
dir_config = Path("/opt/ml/workspace/RecBole/config")
name_dataset = CONFIG['name_dataset']

config_dict = {
    'epochs': 300,
    # 'train_batch_size': 64,
    # 'eval_batch_size': 256,
}
config_file_list = ['whiskey_pairwise.yaml', 'common.yaml']
# config_file_list = ['whiskey_pointwise.yaml', 'common.yaml']

config_file_list = [dir_config / i for i in config_file_list]

# %%
# https://recbole.io/docs/user_guide/model_intro.html
run_recbole(dataset=name_dataset, model="DeepFM", config_file_list=config_file_list, config_dict=config_dict)

# %%
