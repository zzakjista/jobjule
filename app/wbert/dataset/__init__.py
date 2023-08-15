from .base import make_dataset

def dataset_factory(args):
    dataset = make_dataset(args)
    return dataset
