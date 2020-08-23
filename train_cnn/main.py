'''
Things to note for myself (Satoshi)
- this is a bit different from the train_cnn_pytorch in toyroom-rec
- the dataset txt does not start the category index 0.
- therefore we cannot use e.g. toyroom-rec/data/testing_data/test_processed_images.txt as text txt
'''

import os
import sys
import time
from datetime import datetime
import argparse
from copy import deepcopy
import glob
import pandas as pd

try:
    if not os.environ.get("DISABLE_TQDM"):
        from tqdm import tqdm
        tqdm = tqdm
    else:
        print("progress bar is disabled")
except:
    print("can't import tqdm. progress bar is disabled")
    tqdm = lambda x: x

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

##setup imagebackend
from torchvision import get_image_backend,set_image_backend
try:
    import accimage
    set_image_backend("accimage")
except:
    print("accimage is not available")
print("image backend: %s"%get_image_backend())

# imports from my own script
import utils
utils.make_deterministic(123)
from datasets.ImagePandasDataset import ImagePandasDataset 
from metrics.AverageMeter import AverageMeter
from metrics.accuracy import accuracy

import numpy as np
import random
import json

def setup_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--header', type=int, default=1, help = "txt has hearder or not")
    parser.add_argument('--img-idx', type=int, default=0, help = "index of the column of the img")
    parser.add_argument('--cls-idx', type=int, default=1, help = "index of the column of the label")
    parser.add_argument('--w', type=int, default=224, help = "width")
    parser.add_argument('--h', type=int, default=224, help = "height")
    
    parser.add_argument('--train', type=str, required=True, help = "train dataset txt")
    parser.add_argument('--val', type=str,default = "../data/test_images/test.txt", help = "val dataset txt")
    parser.add_argument('--test', type=str,default = "../data/test_images/test.txt", help = "test dataset txt")
    parser.add_argument('--train-img-root', type=str, default="../data/naming_3s_imgs", help="training image root")
    parser.add_argument('--val-img-root', type=str, default="../data/test_images/", help="validation image root")
    parser.add_argument('--test-img-root', type=str, default="../data/test_images/", help="testing image root")

    parser.add_argument('--workers', type=int, default=8, help="number of processes to make batch worker.")
    parser.add_argument('--gpu','-g', type=int, default=-2,help = "gpu id. -1 means cpu. -2 means use CUDA_VISIBLE_DEVICES one")

    parser.add_argument('--resume', type=str, default=None, help="checkpoint to resume")
    parser.add_argument('--resume-optimizer', type=str, default=None, help="optimizer checkpoint to resume")
    parser.add_argument('--optimizer', type=str, default="sgd",choices = ["adam","sgd"], help = "optmizer")
    parser.add_argument('--lr', type=float, default=0.01, help = "initial learning rate")
    parser.add_argument('--patience', type=int, default=2, help="patience")
    parser.add_argument('--step-facter', type=float, default=0.1, help="facter to decrease learning rate when val loss stop decreasing")
    parser.add_argument('--lr-min', type=float, default=0.0001, help = "if lr becomes less than this, stop")
    parser.add_argument('--batch', type=int, default=128, help="batch size")
    parser.add_argument('--epochs', type=int, default=300, help="maximum number of epochs. if 0, evaluation only mode")

    parser.add_argument('--backbone', type=str,default = "resnet50",choices = ["vgg16","resnet18","resnet50","resnet152"], help = "feature extraction cnn")
    parser.add_argument('--backbone-pretrained', type=int,default = 1, help = "use pretrained model or not")    
    parser.add_argument('--saveroot',  default = "../experiments/default/", help='Root directory to make the output directory')
    parser.add_argument('--saveprefix',  default = "log", help='prefix to append to the name of log directory')
    parser.add_argument('--saveargs',default = ["backbone"],nargs='+', help='args to append to the name of log directory')
    parser.add_argument('--savemodel',  default = 1, type=int, help='save model weights or not')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--loss-balanced',  default = 0, type=int, help='balanced weighted loss or not')
    parser.add_argument('--seed', default=-1, type=int, help='seed. -1 means random from time. -2 is SATOSHI_SEED env')
    
    return parser.parse_args()

def setup_dataset(args):
    #setup dataset as pandas data frame
    df_dict = {}
    df_dict["train"] = pd.read_csv(args.train, sep=' ')
    df_dict["val"] = pd.read_csv(args.val, sep=' ')
    df_dict["test"] = pd.read_csv(args.test, sep=' ')
    
    img_root_dict = {
                    "train":args.train_img_root,
                    "val":args.val_img_root,
                    "test":args.test_img_root,
                    }
    
    df = df_dict["train"]
    target_transform = {label:i for i,label in enumerate(sorted(df[df.columns[args.cls_idx]].unique()))}
    
    dataset_dict = {}
    #key is train/val/test and the value is corresponding pytorch dataset
    for split,df in df_dict.items():
        #target_transform is mapping from category name to category idx start from 0
        if split=="train":
            transform = transforms.Compose([
                transforms.Resize((args.h,args.w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((args.h,args.w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        dataset_dict[split] = ImagePandasDataset(df=df,
                                            img_key=df.columns[args.img_idx],
                                            label_key = df.columns[args.cls_idx],
                                            transform = transform,
                                            target_transform = target_transform,
                                            img_root = img_root_dict[split],
                                            )
    return dataset_dict

def setup_dataloader(args,dataset_dic):
    dataloader_dict = {}
    for split,dataset in dataset_dic.items():
        dataloader_dict[split] = DataLoader(dataset,
                                          batch_size=args.batch, 
                                          shuffle= split=="train",
                                          num_workers=args.workers,
                                          pin_memory=True,
                                         )
    return dataloader_dict

def setup_backbone(name,pretrained=False,num_classes=12):
    if name == "vgg16":
        model = torchvision.models.vgg16(pretrained=pretrained)    
        num_features = int(model.classifier[6].in_features)
        model.classifier[6] = nn.Linear(num_features,num_classes)
        return model
    elif name == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)    
        num_features = int(model.fc.in_features)
        model.fc = nn.Linear(num_features,num_classes)
        return model
    elif name == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)    
        num_features = int(model.fc.in_features)
        model.fc = nn.Linear(num_features,num_classes)
        return model
    elif name == "resnet152":
        model = torchvision.models.resnet152(pretrained=pretrained)    
        num_features = int(model.fc.in_features)
        model.fc = nn.Linear(num_features,num_classes)
        return model
    else:
        raise NotImplementedError("this option is not defined")

def train_one_epoch(dataloader,model,criterion,optimizer,accuracy=accuracy,device=None,print_freq=100):
    since = time.time()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train() # Set model to training mode
    
    losses = AverageMeter()
    accs = AverageMeter()
   
    for i,data in enumerate(tqdm(dataloader)):
        inputs = data["input"].to(device)
        labels = data["label"].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), outputs.size(0))
        accs.update(acc.item(),outputs.size(0))

        if i % print_freq == 0 or i == len(dataloader)-1:
            temp = "current loss: %0.5f "%loss.item()
            temp += "acc %0.5f "%acc.item()
            temp += "| running average loss %0.5f "%losses.avg
            temp += "acc %0.5f "%accs.avg
            print(i,temp)

    time_elapsed = time.time() - since
    print('this epoch took {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return float(losses.avg),float(accs.avg)

def evaluate(dataloader,model,criterion,accuracy,device=None):        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    with torch.no_grad():
        for i,data in enumerate(tqdm(dataloader)):
            inputs = data["input"].to(device)
            labels = data["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            losses.update(loss.item(), outputs.size(0))
            accs.update(acc.item(),outputs.size(0))

    print("eval loss %0.5f acc %0.5f "%(losses.avg,accs.avg))    
    return float(losses.avg),float(accs.avg)

def main(args):
    since = time.time()
    print(args)
    if args.seed < 0:
        if os.getenv('SATOSHI_SEED') is not None and args.seed == -2:
            args.seed = int(os.getenv('SATOSHI_SEED'))
            print("env seed used")
        else:
            import math
            seed = int(10**4*math.modf(time.time())[0])
            args.seed = seed
    print("random seed",args.seed)
    utils.make_deterministic(args.seed)
        
    #setup the directory to save the experiment log and trained models
    log_dir =  utils.setup_savedir(prefix=args.saveprefix,basedir=args.saveroot,args=args,
                                   append_args = args.saveargs)
    
    #save args
    utils.save_args(log_dir,args)
    
    #setup gpu
    if args.gpu==-2 and os.getenv('CUDA_VISIBLE_DEVICES') is not None:
        args.gpu = os.getenv('CUDA_VISIBLE_DEVICES')
        
    device = utils.setup_device(args.gpu)
        
    #setup dataset and dataloaders
    dataset_dict = setup_dataset(args)
    dataloader_dict = setup_dataloader(args,dataset_dict)
        
    #setup backbone cnn
    num_classes = dataset_dict["train"].num_classes
    model = setup_backbone(args.backbone,pretrained = args.backbone_pretrained,num_classes=num_classes)

    #resume model if needed
    if args.resume is not None:
        model = utils.resume_model(model,args.resume,state_dict_key = "model")

    #setup loss
    criterion = torch.nn.CrossEntropyLoss().to(device)
    if args.loss_balanced:
        print("using balanced loss")
        #if this optin is true, weight the loss inversely proportional to class frequency
        weight = torch.FloatTensor(dataset_dict["train"].inverse_label_freq)
        criterion = torch.nn.CrossEntropyLoss(weight=weight).to(device)

    #setup optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,amsgrad=True)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr)
    else:
        raise NotImplementedError()
    if args.resume_optimizer is not None:
        optimizer = utils.resume_model(optimizer,args.resume_optimizer,state_dict_key = "optimizer")
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=args.patience, factor=args.step_facter,verbose=True)
    
    #main training
    log = {}
    log["git"] = utils.check_gitstatus()
    log["timestamp"] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log["train"] = []
    log["val"] = []
    log["lr"] = []
    log_save_path = os.path.join(log_dir,"log.json")
    utils.save_json(log,log_save_path)
    valacc = 0
    best_val_acc = 0
    bestmodel = model
    for epoch in range(args.epochs):
        print("epoch: %d --start from 0 and at most end at %d"%(epoch,args.epochs-1))
        loss,acc = train_one_epoch(dataloader_dict["train"],model,criterion,
                        optimizer,accuracy=accuracy,
                        device=device,print_freq=args.print_freq)
        log["train"].append({'epoch':epoch,"loss":loss,"acc":acc})
        
        valloss,valacc = evaluate(dataloader_dict["val"],model,criterion,accuracy=accuracy,device=device)
        log["val"].append({'epoch':epoch,"loss":valloss,"acc":valacc})
        lr_scheduler.step(valloss)
        
        #if this is the best model so far, keep it on cpu and save it
        if valacc > best_val_acc:
            best_val_acc = valacc
            log["best_epoch"] = epoch
            log["best_acc"] = best_val_acc
            bestmodel = deepcopy(model)
            bestmodel.cpu()
            if args.savemodel:
                save_path = os.path.join(log_dir,"bestmodel.pth")
                utils.save_checkpoint(save_path,bestmodel,key="model")
                save_path = os.path.join(log_dir,"bestmodel_optimizer.pth")
                utils.save_checkpoint(save_path,optimizer,key="optimizer")

            
        utils.save_json(log,log_save_path)
        max_lr_now = max([ group['lr'] for group in optimizer.param_groups ])
        log["lr"].append(max_lr_now)
        if max_lr_now < args.lr_min:
            break
            
    #use the best model to evaluate on test set
    print("test started")
    loss,acc  = evaluate(dataloader_dict["test"],bestmodel,criterion,accuracy=accuracy,device=device)
    log["test"] = {"loss":loss,"acc":acc}
    
    time_elapsed = time.time() - since
    log["time_elapsed"] = time_elapsed
    #save the final log
    utils.save_json(log,log_save_path)
    
if __name__ == '__main__':
    args = setup_args()
    main(args)