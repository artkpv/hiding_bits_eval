# %%
import os
from datasets import load_dataset, Dataset, DatasetDict

from functools import partial
from Context import Context

SEED = 100500  # TODO make one seed for all tasks

from utils import map_fn


# %%
def build_dataset(context: Context) -> DatasetDict:
    imdb: DatasetDict = load_dataset("imdb", split="train+test")  # type: ignore

    # Get num_proc from current number of cores div by 2:
    num_proc = os.cpu_count() // 2  # type: ignore
    ds = (
        imdb.shuffle(SEED)
        .select(range(context.ds_size))  # type: ignore
        .map(
            partial(map_fn, context=context),
            load_from_cache_file=False,
            #batched=True,
            #num_proc=num_proc,
        )
    )
    ds.set_format(type="torch")
    ds = ds.train_test_split(test_size=context.train_test_split, seed=SEED)
    return ds


# %%

if __name__ == "__main__":
    ds = build_dataset(TrainContext(ds_size=100))
# %%
