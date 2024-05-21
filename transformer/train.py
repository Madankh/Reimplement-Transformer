import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, random_split, DataLoader 

from dataset import BilingualDataset, causal_mask

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config,ds , lang):
    tokenizer_Path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_Path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_Path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_Path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config['lang_tgt']}', spilt='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config,ds_raw,config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config,ds_raw,config["lang_tgt"])

    # Keep 90% data for traning and 10% for validation
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw , [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw,tokenizer_src, tokenizer_tgt , config['src_lang'], config['tgt_lang' , config['seq_len']])
    val_ds = BilingualDataset(val_ds_raw,tokenizer_src, tokenizer_tgt , config['src_lang'], config['tgt_lang' , config['seq_len']])



