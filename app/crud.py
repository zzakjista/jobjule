import pandas as pd 
import math
from wbert.path import path_controller
import json

class CRUD:

    def __init__(self):
        self.list_num = 10
        self.export_path = path_controller()
        
    def get_recommend_item(self, id):
        recommendation_result_path = self.export_path._get_recommendation_result_path()
        with open(f'wbert/{recommendation_result_path}', "r") as f:
            all_recommendation_result = json.load(f)
        recommendation_result = all_recommendation_result[id]
        return recommendation_result
    
    def get_item(self):
        job_simple_path, job_specific_path = self.export_path._get_rawdata_datasets_path()
        job_simple = pd.read_csv(f'wbert/{job_simple_path}')
        job_specific = pd.read_csv(f'wbert/{job_specific_path}')
        _item_list = pd.merge(job_simple, job_specific, on='구인인증번호', how='left')
        total_count = _item_list['구인인증번호'].count()
        return total_count, _item_list

    def get_page(self,page):
        total, _item_list = self.get_item()
        total_page = 1
        start_index = (page-1)*self.list_num
        end_index = page*self.list_num
        _item_list = _item_list.iloc[start_index:end_index]
        return total_page, _item_list
