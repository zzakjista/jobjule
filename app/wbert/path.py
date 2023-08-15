from pathlib import Path
from datetime import datetime


class path_controller:

    def __init__(self):
        self.today = datetime.today().strftime('%Y%m%d')
    
    # save dataset path # 
    def _get_data_root_path(self):
        return Path('data')

    def _get_rawdata_root_path(self):
        root = self._get_data_root_path()
        return root.joinpath('raw_data')
    
    def _get_geosite_datasets_path(self):
        folder = self._get_data_root_path()
        return folder.joinpath('geo_site.csv')

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        folder_name = 'date_{}'.format(self.today)
        return root.joinpath(folder_name)
    
    def _get_rawdata_datasets_path(self):
        folder = self._get_rawdata_folder_path()
        job_simple_path = folder.joinpath('채용목록.csv')
        job_specific_path = folder.joinpath('채용상세.csv')
        return job_simple_path , job_specific_path
    
    def _get_preprocessed_root_path(self):
        root = self._get_data_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = 'date_{}'.format(self.today) 
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        dataset = folder.joinpath('dataset.pkl')
        return dataset
    
    def _get_preprocessed_recommend_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        dataset = folder.joinpath('recommend_dataset.pkl')
        return dataset
    

    # save model path #
    def _get_model_root_path(self):
        return Path('model_result')
    
    def _get_version_folder_path(self):
        root = self._get_model_root_path()
        folder_name = 'date_{}'.format(self.today)
        return root.joinpath(folder_name)
    
    def _get_recent_model_path(self):
        folder = self._get_version_folder_path()
        return folder.joinpath('recent_model.pth')
    
    def _get_best_acc_model_path(self):
        folder = self._get_version_folder_path()
        return folder.joinpath('best_acc_model.pth')
    
    
    # save recommendation path #
    def _get_recommendation_root_path(self):
        return Path('recommendation_result')
    
    def _get_recommendation_folder_path(self):
        root = self._get_recommendation_root_path()
        folder_name = 'date_{}'.format(self.today)
        return root.joinpath(folder_name)
    
    def _get_recommendation_result_path(self):
        folder = self._get_recommendation_folder_path()
        return folder.joinpath('recommend.json')

