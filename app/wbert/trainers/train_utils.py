import torch

#파일명 변경 시 templates.py 및 path.py의 내용을 변경해야합니다.
def save_recent_model(model, optimizer, epoch, version_path, val_acc, file_name = 'recent_model.pth'): 
    
    model_path = version_path.joinpath(file_name)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    torch.save(state, model_path)
    print(f"Model saved to {model_path} at epoch {epoch}")
    return val_acc

def save_best_model(model, optimizer, epoch, version_path, val_acc, best_val_acc, file_name = 'best_acc_model.pth'):
    
    model_path = version_path.joinpath(file_name)
    if val_acc > best_val_acc:
        print(f"Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving checkpoint...")
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': val_acc
        }
        torch.save(state, model_path)
        print(f"Checkpoint saved to {model_path} at epoch {epoch}")
        return val_acc
    else:
        print(f"Checkpoint not saved. Current validation accuracy {val_acc:.4f} did not improve from best {best_val_acc:.4f}.")
        return best_val_acc