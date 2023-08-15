import torch.nn as nn
from transformers import AdamW, get_cosine_schedule_with_warmup
import torch
from tqdm import tqdm_notebook
from .train_utils import *


class BERTTrainer:

    def __init__(self,args, model, train_loader, val_loader, test_loader, export_path):
        self.device = args.device
        self.model = model.to(self.device)
        

        self.is_parallel = args.num_gpu > 1
        if self.is_parallel:
            self.model = nn.DataParallel(self.model) # GPU를 병렬로 사용 시 작동
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.version_path = export_path._get_version_folder_path()

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        no_decay = ['bert.bias', 'bert.LayerNorm.weight'] # 가중치 변경이 되지 않는 파라미터 집합
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        if args.model_mode =='fine_tuned' and args.load_optimizer:
            print('loading optmizer state dict...')
            optim_path = export_path._get_model_root_path().joinpath('date_{}'.format(args.model_version)).joinpath(args.model_path)
            self.optimizer.load_state_dict(torch.load(optim_path, map_location=torch.device(self.device)).get('optimizer_state_dict'))
            print('complete optmizer state dict...')

        self.num_epochs = args.num_epochs
        self.log_interval = args.log_interval
        self.max_grad_norm = args.max_grad_norm
        self.loss_fn = nn.CrossEntropyLoss()
        self.warmup_ratio = args.warmup_ratio
        self.t_total = int(len(self.train_loader) * self.num_epochs)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.t_total*self.warmup_ratio, num_training_steps=self.t_total)
        self.enable_scheduler = args.enable_scheduler
        self.best_val_acc = 0.0


    def train(self):
        self.validate(0)
        self.model.train()
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            self.validate(epoch)

    def train_one_epoch(self,epoch):
        train_acc = 0.0
        if self.enable_scheduler:
            self.scheduler.step()
        self.model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(self.train_loader)):
            self.optimizer.zero_grad()
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length= valid_length
            label = label.long().to(self.device)
            out = self.model(token_ids, valid_length, segment_ids)
            loss = self.loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            train_acc += self.calc_accuracy(out, label)
            if batch_id % self.log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(epoch+1, batch_id+1, loss, train_acc / (batch_id+1)))
        print("epoch {} train acc {}".format(epoch+1, train_acc / (batch_id+1)))
    

    def validate(self, epoch):
        val_acc = 0.0
        self.model.eval() 
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(self.val_loader)): #test
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length= valid_length
                label = label.long().to(self.device)
                out = self.model(token_ids, valid_length, segment_ids)
                val_acc += self.calc_accuracy(out, label)
        print( "validation acc {}".format( val_acc / (batch_id+1)))
        save_recent_model(self.model, self.optimizer, epoch+1, self.version_path, val_acc / (batch_id+1))
        self.best_val_acc = save_best_model(self.model, self.optimizer, epoch+1, self.version_path, val_acc / (batch_id+1), self.best_val_acc)
    

    def test(self):
        best_model_path = self.version_path.joinpath('best_acc_model.pth')
        best_model = torch.load(best_model_path).get('model_state_dict')
        self.model.load_state_dict(best_model)

        self.model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(self.test_loader)):
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length= valid_length
                label = label.long().to(self.device)
                out = self.model(token_ids, valid_length, segment_ids)
                test_acc += self.calc_accuracy(out, label)
            print("test acc {}".format(test_acc / (batch_id+1)))
    
    def calc_accuracy(self, X, Y):
        max_vals, max_indices = torch.max(X, 1)
        train_acc = (max_indices == Y).sum() / max_indices.size()[0]
        return train_acc
