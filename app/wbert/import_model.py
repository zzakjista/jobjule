import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
import gluonnlp as nlp

class pretrained_model:

    def __init__(self, args):
        self.vocab = None
        self.model = None
        self.tokenizer = None
        self.model_mode = args.model_mode
        
        # 로컬로 모델을 다운받아 실행할 경우 경로를 변경합니다 # 
    def import_model(self):
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        self.model = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
        self.vocab = nlp.vocab.BERTVocab.from_sentencepiece(self.tokenizer.vocab_file, padding_token='[PAD]')

            