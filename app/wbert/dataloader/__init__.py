from dataset import dataset_factory
from .bert import BertDataloader

def dataloader_factory(args, tokenizer, vocab):
    dataset = dataset_factory(args)
    dataloader = BertDataloader(args, tokenizer, vocab, dataset)
    train, val, test = dataloader.get_dataloaders()
    return train, val, test
