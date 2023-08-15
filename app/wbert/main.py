from options import args
from path import path_controller
from collection import collector
from dataloader import dataloader_factory
from trainers import trainer_factory
from models import model_factory
from recommendation import start_recommendation
from utils import *
import pandas as pd


def train():
    export_path = path_controller()
    setup_train(args,export_path)
    model, tokenizer, vocab = model_factory(args,export_path)
    train_loader, val_loader, test_loader = dataloader_factory(args, tokenizer, vocab)
    trainer = trainer_factory(args,model, train_loader, val_loader, test_loader, export_path)
    trainer.train()
    if input('Do you want to test the model? (y/n)') == 'y':
        trainer.test()
    return

def recommend(args): # 
    export_path = path_controller()
    pth = setup_recommend(args,export_path)
    model, tokenizer, vocab = model_factory(args,export_path)
    recommend_dictionary = start_recommendation(args, model, tokenizer, export_path)
    with open(f'{pth}/recommend.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(recommend_dictionary, ensure_ascii=False, indent='\t'))
    print('done')


if __name__ == '__main__':
    collector(args)
    print(args.mode, 'is running')
    if args.mode == 'train':
        train()
    elif args.mode == 'recommend':
        recommend(args)
