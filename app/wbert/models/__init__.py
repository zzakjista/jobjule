import torch
from import_model import pretrained_model
from .bert_classifier import BERTClassifier


def model_factory(args,path_controller):
    pretrained = pretrained_model(args)
    pretrained.import_model()
    model = pretrained.model
    tokenizer = pretrained.tokenizer
    vocab = pretrained.vocab
    model = BERTClassifier(model,args)
    print('model is loaded')
    if args.model_mode == 'fine_tuned':
        print('fine tuned model is loading...')
        model.load_state_dict(_get_state_dict(args,path_controller),strict=False) 
        # model 파라미터의 이름이 다를 경우 같은 이름만 매칭하여 파라미터를 업데이트합니다. #
        print('fine tuned model is loaded')
    return model, tokenizer, vocab

def _get_state_dict(args,path_controller):
    model_version_path = path_controller._get_model_root_path().joinpath('date_{}'.format(args.model_version))
    model_path = model_version_path.joinpath(args.model_path)
    return torch.load(model_path, map_location=torch.device(args.device)).get('model_state_dict')