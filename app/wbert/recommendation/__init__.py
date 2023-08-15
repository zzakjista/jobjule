from dataset import dataset_factory
from .recommend import recommender
from .cascade_filter import cascade_filtering

def start_recommendation(args, model, tokenizer, export_path):
    dataset = dataset_factory(args).load_dataset()
    print('dataset is loaded')
    recommend_agent = recommender(args, model, tokenizer, dataset)
    print('getting recommend dictionary...')
    recommend_dictionary = recommend_agent.get_recommend_dictionary()
    print('recommend dictionary is loaded')
    filtering_agent = cascade_filtering(args, recommend_dictionary, export_path)
    print('start filtering...')
    recommend_dictionary = filtering_agent.filtering()
    print('filtering is done')
    return recommend_dictionary
    
