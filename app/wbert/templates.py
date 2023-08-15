def set_template(args):

    args.item_num = 20
    args.service_key = 'type your service key'
    if args.template is None:
        return
    
    elif args.template.startswith('train'):
        args.mode = 'train'

        args.device = 'cuda' # 'cpu'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'AdamW'
        args.enable_scheduler = True
        args.lr = 0.00005
        args.weight_decay = 0.01
        args.momentum = None
        args.num_epochs = 5
        batch = 32
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch
        args.model_init_seed = 0
        args.max_len = 256
        args.hidden_units = 768
        args.dr_rate = 0.1
        args.model_mode = 'original'
        args.model_version = '00000000'
        args.model_path = 'best_acc_model.pth'
        args.load_optimizer = False # 학습 중단 시 True로 변경
        args.sent_idx = 0
        args.label_idx = 1
        
    
    elif args.template.startswith('recommend'):
        args.mode = 'recommend'

        args.model_mode = 'original'
        args.model_version = '00000000'
        args.model_path = 'best_acc_model.pth'
        
        args.nearest_k = 1000
        args.sudo_dist = 20
        args.province_dist = 40
        return

# fine_tuned 모델 파라미터 사용 예제 #
# args.model_model = 'fine_tuned'
# args.model_version = '00000000'
# args.model_path = 'best_acc_model.pth'