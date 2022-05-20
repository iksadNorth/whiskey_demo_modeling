# %%
from anyio import open_file
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_scores, full_sort_topk
from pathlib import Path
from glob import glob
from shutil import copytree, ignore_patterns

import yaml
with open('config.yaml') as f:
    CONFIG = yaml.load(f)

# %%
dir_model_saved = Path(CONFIG['dir_model_saved'])
file_model_used = CONFIG['file_model_used']

# %%
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=dir_model_saved / file_model_used,
)

uid_series = dataset.token2id(dataset.uid_field, ['whiskycuse', 'Nopax'])
# uid_series = dataset.token2id(dataset.uid_field, 'Nopax')

topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
print(topk_score)
print(topk_iid_list)
external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
print(external_item_list)

# %%
