# %%
from recbole.quick_start import run_recbole

from pathlib import Path
from glob import glob
from shutil import copytree, ignore_patterns


# %%
dir_config = Path("/opt/ml/workspace/RecBole/config")

config_dict = {
    'epochs': 300,
    # 'train_batch_size': 64,
    # 'eval_batch_size': 256,
}
config_file_list = ['whiskey_pairwise.yaml', 'common.yaml']
# config_file_list = ['whiskey_pointwise.yaml', 'common.yaml']

config_file_list = [dir_config / i for i in config_file_list]

# %%
run_recbole(dataset="whiskey", model="ConvNCF", config_file_list=config_file_list, config_dict=config_dict)

# %%
