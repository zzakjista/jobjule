from .train_model import BERTTrainer

def trainer_factory(args, model, train_loader, val_loader, test_loader, export_path):
    return BERTTrainer(args, model, train_loader, val_loader, test_loader, export_path)
    