import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 4 

if __name__ == '__main__':
    print("Loading TinyStories (only a small part)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train[:5%]")
    
    enc = tiktoken.get_encoding("gpt2")

    def process(example):
        ids = enc.encode_ordinary(example['text']) 
        ids.append(enc.eot_token) 
        return {'ids': ids, 'len': len(ids)}

    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=num_proc,
    )

    data = np.concatenate([np.array(x) for x in tokenized['ids']])
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    train_data = train_data.astype(np.uint16)
    val_data = val_data.astype(np.uint16)
    
    train_data.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_data.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
    print("Done! You can now start the training.")
