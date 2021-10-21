# %%
from datasets import load_dataset_builder
dataset_builder = load_dataset_builder('imdb')
print(dataset_builder.cache_dir)
print(dataset_builder.info.features)
print(dataset_builder.info.splits)
# %%
from datasets import load_dataset
dataset = load_dataset('imdb', split='train')
dataset
# %%
dataset = dataset.shuffle().train_test_split(0.2)
# %%
dataset['test']
# %%
dataset['test']['text']