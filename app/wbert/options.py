from templates import set_template
import argparse


parser = argparse.ArgumentParser(description='RecPlay')

## main ##
parser.add_argument('--mode', type=str, default=None, choices=['train','test','recommend','visualization'])
parser.add_argument('--template', type=str, default=None)


## collection ##
parser.add_argument('--end_point', type=str, default='http://openapi.work.go.kr/opi/opi/opia/wantedApi.do?')
parser.add_argument('--service_key', type=str, default=None) 
parser.add_argument('--call_tp', type=str, default='L',choices=['L','D'])
parser.add_argument('--return_type', type=str, default='XML')
parser.add_argument('--item_num', type=int, default=1000) 


## Dataset ##
parser.add_argument('--sent_idx', type=int, default=0)
parser.add_argument('--label_idx', type=int, default=1)
parser.add_argument('--pad', type=bool, default=True)
parser.add_argument('--pair', type=bool, default=False)


## DataLoader ##
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)


## Train ##
# Device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')

# optimizer #
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate') 
parser.add_argument('--weight_decay', type=float, default=0.1, help='l2 regularization')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Define warmup step')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='clipping maximum gradient norm')
parser.add_argument('--enable_scheduler', type=bool, default=True, help='whether to use scheduler or not')

# epochs #
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training')
parser.add_argument('--log_interval', type=int, default=200, help='logging accuracy every n iterations')


## Model ##
parser.add_argument('--max_len', type=int, default=128, help='Length of sequence for bert')
parser.add_argument('--hidden_units', type=int, default=768, help='Size of hidden vectors (d_model)') 
parser.add_argument('--dr_rate', type=float, default=0.1, help='Dropout probability to use throughout the model')
parser.add_argument('--num_classes', type=int, default=13, help='Number of classes to predict')


## import model ##
parser.add_argument('--model_mode', type=str, default='original',choices=['original','fine_tuned'])
parser.add_argument('--model_version', type=str, default=None, help='8 digit date format, such as 20230101')
parser.add_argument('--model_path', type=str, default='best_acc_model.pth',choices=['best_acc_model.pth', 'recent_model.pth'])
parser.add_argument('--load_optimizer', type=bool, default=True, help='whether to load optimizer state dict or not')


## recommendation ##
parser.add_argument('--nearest_k', type=int, default=1000, help='Number of nearest neighbors to retrieve')
parser.add_argument('--sudo_dist', type=int, default=20, help='limitation of distance for sudo nearest neighbors')
parser.add_argument('--province_dist', type=int, default=20, help='limitation of distance for province nearest neighbors')




# args = parser.parse_args(args=[]) # namespace 오류를 방지하기 위해 주피터 커널에서 실행 시 활성화합니다
args = parser.parse_args()
set_template(args)
