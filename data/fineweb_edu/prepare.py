import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    tokenized = split_dataset.map(process, remove_columns=['text'], desc="tokenizing", num_proc=8)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.int64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        idx = 0
        for example in tqdm(dset, desc=f"writing {filename}"):
            arr[idx : idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()
