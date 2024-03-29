import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from clip import tokenize
from imagenet_zero_shot_data import imagenet_classnames, openai_imagenet_template

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
    
class Evaluator():
    def __init__(self, model, val_loader):
        super(Evaluator, self).__init__()
        self.model = model
        self.val_loader = val_loader
        
    def evaluate(self, args):
         with torch.no_grad():
             top1, top5, n = 0., 0., 0.
             for images, labels in tqdm(self.val_loader):
                 images = images.to(args.gpu)
                 labels = labels.to(args.gpu)
                 
                 if args.distributed:
                     image_embeddings = self.model.module.encode_images(images)
                 else:
                     image_embeddings = self.model.encode_images(images)
                 
                 image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
                 
                 zeroshot_weights = self._get_zero_shot_weights(args)
                 
                 if args.bregman:
                     if args.distributed:
                         image_scores = self.model.module.bregman(image_embeddings)
                         text_scores = self.model.module.bregman(zeroshot_weights)
                     else:
                         image_scores = self.model.bregman(image_embeddings)
                         text_scores = self.model.bregman(zeroshot_weights)
                     
                     batch_size = image_scores.shape[0]
                     dist_matrix = torch.sub(image_scores.repeat(1, batch_size).T, text_scores)
                     sigma = torch.tensor([args.length_scale]).to(args.gpu)
                     sigma = 2 * torch.pow(sigma, 2)
                     sim_matrix = torch.exp(torch.div(-dist_matrix, sigma))
                     
                 else:
                     sim_matrix = 100. * image_embeddings@zeroshot_weights
                 
                 acc1, acc5 = self._accuracy(sim_matrix, labels, topk=(1, 5))
                 top1 += acc1
                 top5 += acc5
                 n += images.size(0)

             top1 = (top1 / n)
             top5 = (top5 / n)
         return top1, top5
                     
    
    def _get_zero_shot_weights(self, args):
        zeroshot_weights = []
        for classname in imagenet_classnames:
            texts = [template(classname) for template in openai_imagenet_template]
            texts = self.model.tokenize(texts).to(args.gpu)
            
            if args.distributed:
                class_embeddings = self.model.module.encode_text(texts)
            else:
                class_embeddings = self.model.encode_text(texts)
                
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.gpu)
        return zeroshot_weights
    
    def _accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
        
        
        
        