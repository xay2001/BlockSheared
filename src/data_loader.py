import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer, use_local=False, local_path=None):
    if use_local and local_path:
        traindata = load_dataset('text', data_files={'train': f'{local_path}/train.txt'},disable_tqdm=False)['train']
        testdata = load_dataset('text', data_files={'test': f'{local_path}/test.txt'},disable_tqdm=False)['test']
    else:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', disable_tqdm=False)
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', disable_tqdm=False)

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process C4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer, use_local=True, local_train_path=None, local_val_path=None):
    if use_local and local_train_path and local_val_path:
        traindata = load_dataset('json', data_files={'train': local_train_path}, disable_tqdm=False)['train']
        valdata = load_dataset('json', data_files={'validation': local_val_path}, disable_tqdm=False)['validation']
    else:
        traindata = load_dataset('c4', 'en', split='train', disable_tqdm=False)
        valdata = load_dataset('c4', 'en', split='validation', disable_tqdm=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, use_local=False, local_paths=None):
    if 'wikitext2' in name:
        local_path = local_paths.get('wikitext2') if local_paths else None
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, use_local=use_local, local_path=local_path)
    if "c4" in name:
        local_train_path = local_paths.get('c4_train') if local_paths else None
        local_val_path = local_paths.get('c4_val') if local_paths else None
        return get_c4(nsamples, seed, seqlen, tokenizer, use_local=use_local, local_train_path=local_train_path, local_val_path=local_val_path)
