import os 
import json


def setup_train(args, path_controller):
    experiment_path = path_controller._get_version_folder_path()
    experiment_path = create_folder(experiment_path)
    export_experiments_config_as_json(args, experiment_path)

def setup_recommend(args, path_controller):
    recommend_path = path_controller._get_recommendation_folder_path()
    recommend_path = create_folder(recommend_path)
    return recommend_path
    
def create_folder(folder_path):
    if not folder_path.is_dir():
        folder_path.mkdir(parents=True)
    return folder_path

def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)
