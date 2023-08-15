import faiss
from .document_embedding import EmbeddingVectorizer

class recommender(EmbeddingVectorizer):

    def __init__(self,args, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.nearest_k = args.nearest_k
        self.embeddings = self.model.state_dict()['bert.embeddings.word_embeddings.weight']
        self.docu_embeddings = self.get_embeddings()

    def get_recommend_dictionary(self):
        calculator = self._open_calcuator()
        indices = self._calcuate_distance(calculator)
        recommend_dictionary = self._make_recommend_dictionary(indices)
        return recommend_dictionary

    def _make_recommend_dictionary(self, indices):
        recommend_dictionary = {}
        for i in range(len(self.dataset)):
            recommend_dictionary[self.dataset.keys()[i]] = [self.dataset.keys()[j] for j in indices[i]]
            if i % 10000 == 0:
                print(i,'document are recommended')
        print('recommendation finished')
        return recommend_dictionary

    def _calcuate_distance(self,calculator):
        query = self.docu_embeddings
        distances, indices= calculator.search(query, self.nearest_k)
        return indices

    def _open_calcuator(self):
        calculator = faiss.IndexFlatL2(self.docu_embeddings.shape[1])
        calculator.add(self.docu_embeddings)
        return calculator