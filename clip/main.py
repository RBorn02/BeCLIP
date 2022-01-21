import argparse
import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


from data import get_csv_dataloader, get_imagenet_dataloader
from trainer import Trainer
from evaluator import Evaluator
from model import build_model
import clip
from configs import get_config


from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train BeCLIP')
    parser.add_argument('data_train', help='path to training data')
    parser.add_argument('data_val', help='path to validation data')
    parser.add_argument('vocab_file', help='path to vocab file')
    parser.add_argument('--model', default='resnet50', help='vision encoder model',
                        choices=['resnet50', 'resnet101', 'resnet50x4', 'resnet50x16',
                                 'vit16-b', 'vit32-b'])
    parser.add_argument('--breg_dim', default=256, help='bregman network hidden size')
    parser.add_argument('--d_subs', default=500, help='number of bregman networks')
    parser.add_argument('--length', default=0.9, help='length scale parameter of bregman loss')
    parser.add_argument('--lmbda', default=5, help='scaling of bregman and nt xent loss')
    parser.add_argument('--temperature', default=0.2, help='softmax temperature')
    parser.add_argument('--lr', default=5*1e-4, help='learning rate set for resnet50')
    parser.add_argument('--batch_size', default=512, help='batch size for training')
    parser.add_argument('--wd', default=0.2, help='weight decay for adam')
    parser.add_argument('--beta1', default=0.9, help='beta 1 for adam')
    parser.add_argument('--beta2', defualt=0.999, help='beta 2 for adam')
    parser.add_argument('--epsilon', default=1e-8, help='adam epsilon')
    parser.add_argument('--epochs', default=100, help='number of training epochs')
    parser.add_argument('--warmup', default=10000, help='number of linear warmup steps')
    parser.add_argument('--distributed', default=True, help='use distribute training')
    parser.add_argument('--gpu', default='cuda', help='wether to use gpu for training')
    parser.add_argument('--workers', default=64, help='number of workers')

    args = parser.parse_args()
    
    writer = SummaryWriter()
    save_config_file(writer.logdir, args)
    
    #Load Config
    config = get_config(args)
    
    #Load Training Data and Validation Data
    preprocess_train = clip._transform(config['image_resolution'])
    train_loader = get_csv_dataloader(args, preprocess_train)
    val_loader = get_imagenet_dataloader(args)
    
    #Load Model
    model = build_model(config, args)
    
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]
    
    print('Loading {} model'.format(args.model))
    print(config)
    
    #Initialize Optimizer
    optimizer = optim.Adam([
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},], 
                           lr=args.lr, betas=(args.beta1, args.beta2),
                           weight_decay=args.wd, eps=args.epsilon)
    
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup,
                    max_epochs=args.epochs)
    
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                        device_ids=[range(torch.cuda.device_count())])
        
    #Initialize Trainer and Evaluator
    trainer = Trainer(model, optimizer, scheduler, args.temperature, args.length_scale,
                      args.epochs, args.device, bregman=args.bregman)
    
    evaluator = Evaluator(model, val_loader)
    
    #Training Loop
    results = {'train_loss': [],
               'test_acc@1': [],
               'test_acc@5': [],
              }
    save_name_pre = '{}_K{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.model, args.d_subs,
        args.encoder_dim, args.lr,
        args.breg_dim, args.temperature,
        args.length, args.lmbda, args.epochs)
    csv_dir = os.path.join(writer.log_dir, '{}_stats.csv'.format(save_name_pre))
    model_dir = os.path.join(writer.log_dir, '{}_model.pth'.format(save_name_pre))
    final_model_dir = os.path.join(writer.log_dir, '{}_final_model.pth'.format(save_name_pre))
    fig_dir = os.path.join(writer.log_dir, '{}_loss_acc.png'.format(save_name_pre))
    
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        loss = trainer.train(train_loader, epoch, args.lmbda)
        
        #Clip logit scale at ln(100) like in the paper
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, torch.log(100))
        
        results['train_loss'] = loss
        writer.add_scalar('loss/train', results['train_loss'][-1], epoch)
        
        top1, top5 = evaluator.evaluate(args)
        results['test_acc@1'] = top1
        results['test_acc@5'] = top5
        writer.add_scalar('acc@1/test', results['test_acc@1'][-1], epoch)
        writer.add_scalar('acc@5/test', results['test_acc@5'][-1], epoch)
        
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(csv_dir, index_label='epoch')
        
        if top1 > best_acc:
            best_acc = top1
            if isinstance(model, nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, model_dir)

        if isinstance(model, nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, final_model_dir)
        
    # plotting loss and accuracies
    df = pd.read_csv(csv_dir)
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(20,10))
    axes[0, 0].set_title('Loss/Train')
    axes[0, 1].set_title('acc@1/test')
    axes[1, 1].set_title('acc@5/test')
    sns.lineplot(ax=axes[0, 0], x="epoch", y="train_loss", data=df)
    sns.lineplot(ax=axes[0, 1], x="epoch", y="test_acc@1", data=df)
    sns.lineplot(ax=axes[1, 1], x="epoch", y="test_acc@5", data=df)        
    fig.savefig(fig_dir)
        
        
        
        
    