import torch
import numpy as np


class EmbeddingVectorizer:
    
    def __init__(self,model,tokenizer,dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.embeddings = self.model.state_dict()['bert.embeddings.word_embeddings.weight']

    def get_embeddings(self): 
        arr = []
        for i in range(len(self.dataset)):
            tmp = self.embeddings[self.tokenizer(self.dataset[i])['input_ids']]
            arr.append(np.array(tmp.mean(axis=0)))
            if i % 10000 == 0:
                print(i,'document are processed')
        print(np.array(arr).shape)
        return np.array(arr)



    # 모델을 통한 임베딩 추출 시 사용하는 코드 입니다 # 
    def get_document_embedding_for_model(self):
        document_embedding = {}
        for id in self.dataset.keys():
            document_embedding[id] = self.get_sentence_embedding(self.dataset[id])
            print(len(document_embedding))
        return document_embedding
    
    def get_sentence_embedding(self, sentence):
        tokens = self.tokenize_sentence(sentence)
        tokens = tokens[:500] #kobert max_length is 512 
        embedding_vectors = self.import_embedding_vectors(tokens)
        sentence_embedding = self.aggregate_embedding_vectors(embedding_vectors)
        return sentence_embedding

    # 문서를 토큰화하여 토큰 리스트 반환
    def tokenize_sentence(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        return tokens

    # 토큰 리스트에 해당하는 임베딩 벡터 불러오기
    def import_embedding_vectors(self, tokens):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens) # 토큰 -> 정수 index
        input_ids = self.tokenizer.prepare_for_model(token_ids).input_ids # token_ids를 BERT 모델 포맷에 맞춤
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        with torch.no_grad():
            model_output = self.model(input_ids)
        last_hidden_state = model_output[0]
        embedding_vectors = last_hidden_state.squeeze(0).numpy()
        return embedding_vectors

    # 임베딩 벡터들을 하나의 문서로 집계 (평균)
    def aggregate_embedding_vectors(self, embedding_vectors):
        if len(embedding_vectors) == 0:
            return None
        return np.mean(embedding_vectors, axis=0)