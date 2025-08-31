import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import *
from tqdm import tqdm

from utils.metrics import *
# from zmq import device

from .coattention import *
from .layers import *


class Trainer_AL():
    def __init__(self,
                model, 
                 device,
                 lr,
                 dropout,
                 weight_decay,
                 save_param_path,
                 epoch_stop,
                 epoches,
                 model_name, 
                 save_threshold = 0.6, 
                 start_epoch = 0,
                 ):
        
        self.model = model

        self.device = device
        self.model_name = model_name

        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path= save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
    
        self.criterion = nn.CrossEntropyLoss()
          
        self.beta_selecting=0.1
        self.beta_editing=2

    def train(self,initckp,dataloader_train,dataloader_val):

        since = time.time()

        self.model.cuda()
        self.model.load_state_dict(initckp,strict=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_model_wts_val = copy.deepcopy(self.model.state_dict())
        best_f1_val = 0.0
        best_epoch_val = 0
        is_earlystop = False


        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)
            
            #training phase
            self.model.train()
            print('-' * 10)
            print ('TRAIN')
            print('-' * 10)
            running_loss = 0.0
            
            tpred = []
            tlabel = []
            
            for batch in tqdm(dataloader_train):
                self.optimizer.zero_grad()
                batch_data=batch
                for k,v in batch_data.items():
                    if k!='vid':
                        batch_data[k]=v.cuda()
                labels = batch_data['label']
                tlabel.extend(labels.detach().cpu().numpy().tolist())
                
                outputs,fea= self.model(**batch_data)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                tpred.extend(preds.detach().cpu().numpy().tolist())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                running_loss += loss.item() * labels.size(0)

                
                
            epoch_loss = running_loss / len(dataloader_train.dataset)
            print('Loss: {:.4f} '.format(epoch_loss))
            results = metrics(tlabel, tpred)
            print (results)
            

            #validation phase
            self.model.eval()
            print('-' * 10)
            print ('VAL')
            print('-' * 10)
            val_loss = 0.0
            val_loss_fnd = 0.0
            vpred = []
            vlabel = []
            
            for batch in tqdm(dataloader_val):
                self.optimizer.zero_grad()
                batch_data=batch
                for k,v in batch_data.items():
                    if k!='vid':
                        batch_data[k]=v.cuda()
                labels = batch_data['label']
                vlabel.extend(labels.detach().cpu().numpy().tolist())
                
                outputs,fea= self.model(**batch_data)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                vpred.extend(preds.detach().cpu().numpy().tolist())
                val_loss += loss.item() * labels.size(0)
                
            epoch_loss_val = val_loss / len(dataloader_val.dataset)
            print('Val Loss: {:.4f} '.format(epoch_loss_val))
            results_val = metrics(vlabel, vpred)
            print (results_val)
            
            if results_val['f1'] > best_f1_val:
                best_f1_val = results_val['f1']
                best_model_wts_val = copy.deepcopy(self.model.state_dict())
                best_epoch_val = epoch+1
                if best_f1_val > self.save_threshold:
                    torch.save(self.model.state_dict(), self.save_param_path +self.model_name+ "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_f1_val))
                    print ("saved " + self.save_param_path+self.model_name + "val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_f1_val) )
            else:
                if epoch-best_epoch_val >= self.epoch_stop-1:
                    os.rename(self.save_param_path +self.model_name+ "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_f1_val), self.save_param_path +self.model_name+ "_val_epoch" + str(best_epoch_val) + "_{0:.4f}".format(best_f1_val)+"_final")
                    is_earlystop = True
                    print ("early stopping...")
        
        for file in os.listdir(self.save_param_path):
            if self.model_name in file and "val_epoch" in file and str(best_epoch_val) not in file and "_final" not in file:
                os.remove(self.save_param_path+file)
        return best_model_wts_val
    
    def refer_pool(self,ckp,dataloader_generation_pool):
        self.model.load_state_dict(ckp,strict=False)
        self.model.cuda()
        self.model.eval()
        all_pool_pred = {} 
        for batch in tqdm(dataloader_generation_pool):
            with torch.no_grad():
                batch_data=batch
                for k,v in batch_data.items():
                    if k!='vid':
                        batch_data[k]=v.cuda()
                outputs,fea= self.model(**batch_data)
                probs = F.softmax(outputs, dim=1)
                entrophy = -torch.sum(probs * torch.log(probs+1e-10), dim=1)
                for bidx, vid in enumerate(batch_data['vid']):
                    all_pool_pred[vid]= entrophy[bidx].item()
        return all_pool_pred

    def refer_cal_fea(self,ckp,dataloader_generation_pool):
        self.model.load_state_dict(ckp,strict=False)
        self.model.cuda()
        self.model.eval()
        all_pool_fea = {} 
        for batch in tqdm(dataloader_generation_pool):
            with torch.no_grad():
                batch_data=batch
                for k,v in batch_data.items():
                    if k!='vid':
                        batch_data[k]=v.cuda()
                outputs,fea= self.model(**batch_data)
                
                for bidx, vid in enumerate(batch_data['vid']):
                    all_pool_fea[vid]= fea[bidx]
        return all_pool_fea



    def test(self,ckp,dataloader_test):

        self.model.load_state_dict(ckp,strict=False)
        self.model.cuda()
        self.model.eval()   

        pred = []
        label = []

        for batch in tqdm(dataloader_test):
            with torch.no_grad(): 
                batch_data=batch
                for k,v in batch_data.items():
                    if k!='vid':
                        batch_data[k]=v.cuda()
                batch_label = batch_data['label']
                outputs,fea= self.model(**batch_data) 

                _, batch_preds = torch.max(outputs, dim=1)


                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_preds.detach().cpu().numpy().tolist())


        print (get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print (metrics(label, pred))

        
        return metrics(label, pred)
    
