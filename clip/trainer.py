import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from loss.breg_loss import BregmanLoss
from nt_xent import NT_Xent
from breg_margin_loss import BregMarginLoss


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Trainer():
    def __init__(self, model,
                 optimizer,
                 scheduler,
                 temperature,
                 length_scale,
                 epochs,
                 device,
                 mixed_loss = True,
                 bregman = True,):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.temperature = temperature
        self.length_scale = length_scale
        self.epochs = epochs
        self.device = device
        self.mixed_loss = mixed_loss
        self.bregman = bregman
        
    def train(self, args, data_loader, epoch, lmbda):
        self.model.train()
        batch_size = data_loader.batch_size
        
        bloss = BregmanLoss(batch_size, self.temperature, self.length_scale, self.case)
        nt_xent = NT_Xent(batch_size, self.temperature)
        
        sampler = data_loader.sampler
        sampler.set_epoch(epoch)
        
        total_loss, total_num, tot_max, train_bar = 0.0, 0, 0, tqdm(data_loader)
        num_max = torch.tensor([0])
        
        for images, texts in train_bar:
            images, texts = images.to(self.device), texts.to(self.device)
            
            #Compute Bregman Scores
            if self.bregman is True:
               
               #Gather bregman scores from torch distributed
               if args.distributed:
                  image_scores, text_scores, image_features, text_features = self.model.module(images, texts)
                   
                  world_size = torch.distributed.world_size()
                  rank = torch.distributed.get_rank()
                  
                  gathered_image_scores = [torch.zeros_like(image_scores) for _ in range(world_size)]
                  gathered_text_scores = [torch.zeros_like(text_scores) for _ in range(world_size)]
               
                  torch.distributed.all_gather(gathered_image_scores, image_scores)
                  torch.distributed.all_gather(gathered_text_scores, text_scores)
               
                  all_image_scores = torch.cat([image_scores] 
                                            + gathered_image_scores[:rank]
                                            + gathered_image_scores[rank+1:])
               
                  all_text_scores = torch.cat([text_scores]
                                            + gathered_text_scores[:rank]
                                            + gathered_text_scores[rank+1:])
                  
                  gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
                  gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
                  
                  torch.distributed.all_gather(gathered_image_features, image_features)
                  torch.distributed.all_gather(gathered_text_features, text_features)
                  
                  all_image_features = torch.cat([image_features]
                                                 + gathered_image_features[:rank]
                                                 + gathered_image_features[rank+1:])
                  
                  all_text_features = torch.cat([text_features]
                                                + gathered_text_features[:rank]
                                                + gathered_text_features[rank+1:])
                  
                  loss, num_max = bloss(all_image_scores, all_text_scores)
                  if self.mixed_loss:
                     nt_xent_loss = nt_xent(all_image_features, all_text_features)
                     loss = loss + lmbda * nt_xent_loss
                  
               
               else:
                  image_scores, text_scores, image_features, text_features = self.model(images, texts) 
                   
                  loss, num_max = bloss(image_scores, text_scores)
                  if self.mixed_loss:
                     nt_xent_loss = nt_xent(image_features, text_features)
                     loss = loss + lmbda * nt_xent_loss
                     
                     
            #Computes Classic CLIP Loss as presented in the paper    
            else:
                
                if args.distributed:
                    image_features, text_features, logit_scale = self.model.module(images, texts)
                    
                    world_size = torch.distributed.world_size()
                    rank = torch.distributed.get_rank()
                    
                    gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
                    gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
                    
                    torch.distributed.all_gather(gathered_image_features, image_features)
                    torch.distributed.all_gather(gathered_text_features, text_features)
                    
                    all_image_features = torch.cat([image_features]
                                                   + gathered_image_features[:rank]
                                                   + gathered_image_features[rank+1:])
                    
                    all_text_features = torch.cat([text_features]
                                                  + gathered_text_features[:rank]
                                                  + gathered_text_features[rank+1:])
                    
                    logit_scale = logit_scale.exp()
                    logits_per_image = logit_scale * all_image_features @ all_text_features.t()
                    logits_per_text = logits_per_image.t()
                
                else:
                   image_features, text_features, logit_scale = self.model(images, texts) 
                   
                   logit_scale = logit_scale.exp()
                   logits_per_image = logit_scale * image_features @ text_features.t()
                   logits_per_text = logits_per_image.t()
                
                
                labels = np.arange(image_features.shape[0])
                image_loss = torch.nn.functional.CrossEntropyLoss(logits_per_image)
                text_loss = torch.nn.functional.CrossEntropyLoss(logits_per_text)
                loss = (image_loss + text_loss) / 2
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            tot_max += num_max
            total_num += batch_size
            total_loss += loss.item() * batch_size
            train_bar.set_description(
                '{}Train{} {}Epoch:{} [{}/{}] {}Loss:{}  {:.4f} {}Active Subs:{} [{}/{}]'
                .format(
                    bcolors.OKCYAN, bcolors.ENDC,
                    bcolors.WARNING, bcolors.ENDC,
                    epoch,
                    self.epochs,
                    bcolors.WARNING, bcolors.ENDC,
                    total_loss / total_num,
                    bcolors.WARNING, bcolors.ENDC,
                    len(torch.where(tot_max>10)[0]),
                    tot_max.shape[0]))

        return total_loss/total_num, 

