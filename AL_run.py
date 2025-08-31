import json
import os
import time
import sys
import random

import torch
from torch.utils.data import DataLoader
from utils.dataloader_gen_AL import *
from utils.AL_Trainer import *
from models.SVFEND import *
import numpy as np
import pandas as pd


def _init_fn(worker_id):
    np.random.seed(2025)

class AL_Run():
    def __init__(self,config):
        self.model_name = config['model_name']

        self.dataset_type = config['dataset_type']

        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        
        self.device = config['device']
        self.lr = config['lr']
        self.save_param_dir = config['path_param']
        self.path_tensorboard = config['path_tensorboard']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']
        self.mode='normal'

        self.dataset_name=config['dataset_name']
        self.init_ckp_path=config['init_ckp_path']
        self.al_itteration=config['al_itteration']
        self.al_pool_size=config['al_pool_size']
        
    def get_dataloader(self,input_vids,append_vids=None):
        
        if self.dataset_type=='simpleSVFEND_tvva_gen':
            dataset=SVFENDDataset_tvva_gen(input_vids=input_vids,dataset_name=self.dataset_name)
            collate_fn=SVFEND_tvva_gen_collate_fn

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn,worker_init_fn=_init_fn)
        return dataloader

    def get_model(self):
        if self.model_name=='SimpleSVFEND_tvva':
            model=SimpleSVFEND_tvva(fea_dim=128,dropout=self.dropout)
       
        return model

    
        

    def main(self):
        
        self.model = self.get_model()
        save_vid_path="./"+self.dataset_name+"_candidate_pools/temporal_train_5fold_trial/"
        if not os.path.exists(save_vid_path):
            save_vid_path = os.makedirs(save_vid_path)
        
        if self.dataset_name=='fakett':
            ori_train_vids=[]
            with open('./data/fakett/data-split/vid_time3_train.txt','r') as f:
                for line in f:
                    ori_train_vids.append(line.strip())
            test=[]
            with open('./data/fakett/data-split/vid_time3_test.txt','r') as f:
                for line in f:
                    test.append(line.strip())
            
            or_val_vids=[]
            with open('./data/fakett/data-split/vid_time3_val.txt','r') as f:
                for line in f:
                    or_val_vids.append(line.strip())
        elif self.dataset_name=='fakesv':
            ori_train_vids=[]
            with open('./data/fakesv/data-split/vid_time3_train.txt','r') as f:
                for line in f:
                    ori_train_vids.append(line.strip())
            test=[]
            with open('./data/fakesv/data-split/vid_time3_test.txt','r') as f:
                for line in f:
                    test.append(line.strip())
            
            or_val_vids=[]
            with open('./data/fakesv/data-split/vid_time3_val.txt','r') as f:
                for line in f:
                    or_val_vids.append(line.strip())
        

        
        shuffled_vids = ori_train_vids.copy()
        random.seed(2025)
        random.shuffle(shuffled_vids)

        
        n = len(shuffled_vids)
        k = 5
        div, remainder = divmod(n, k)

        fold1_vids_pool=[]
        fold2_vids_pool=[]
        fold3_vids_pool=[]
        fold4_vids_pool=[]
        fold5_vids_pool=[]

        
        splits = []
        start = 0
        for i in range(k):
           
            end = start + div + (1 if i < remainder else 0)
            splits.append(shuffled_vids[start:end])
            start = end

        for fold_i in range(k):
            val = splits[fold_i]
            train_vids = []
            for fold_j in range(k):
                if fold_j != fold_i:
                    train_vids.extend(splits[fold_j])
            

            if self.dataset_name=='fakett':
                source_real_ids=[]
                with open('./data/fakett_vids/real_human.txt','r') as f:
                    for line in f:
                        source_real_ids.append(line.strip())

                pd_all_gen=pd.read_json('./data/enhanced_v1_fakett.json',orient='records',lines=True)
            elif self.dataset_name=='fakesv':
                source_real_ids=[]
                with open('./data/fakesv_vids/real_human_train.txt','r') as f:
                    for line in f:
                        source_real_ids.append(line.strip())
                pd_all_gen=pd.read_json('./data/enhanced_v1_fakesv.json',orient='records',lines=True)

            generated_pool=pd_all_gen[pd_all_gen.apply(lambda x: x['source'] != 'Human' and x['video_id'].split('_')[0] not in source_real_ids,axis=1)]['video_id'].tolist()

            initial_train=train_vids

            al_iterations=self.al_itteration
            useful_unvalid_generated_pool=[]
            useless_unvalid_generated_pool=[]

            inital_ckp=torch.load(self.init_ckp_path) if self.init_ckp_path is not None else self.model.state_dict()
            Trainer=Trainer_AL(model=self.model,device=self.device,lr=self.lr,dropout=self.dropout,weight_decay=self.weight_decay,model_name=self.model_name,epoch_stop=self.epoch_stop,save_param_path=self.save_param_dir+self.dataset_name+"/AL_temporal/"+self.model_name+"/"+"poolsize"+str(self.al_pool_size)+"/",epoches=self.epoches)
            dataloader_test=self.get_dataloader(test)
            dataloader_val=self.get_dataloader(val)
            for iteration in range(al_iterations):
                print('iteration:',iteration)
                print('Cur ckp performance on Val:\n')
                cur_ckp_test_metric=Trainer.test(inital_ckp,dataloader_val)
                print('Cur ckp performance on Test:\n')
                _=Trainer.test(inital_ckp,dataloader_test)
                cur_ckp_test_f1=cur_ckp_test_metric['f1']
                # print('pre test f1:',cur_ckp_test_f1)


                val_entropy=Trainer.refer_pool(inital_ckp,dataloader_val)
                anchor_val_samples=[]

                val_entropy=sorted(val_entropy.items(),key=lambda x:x[1],reverse=True)
                anchor_val_samples=[x[0] for x in val_entropy[:int(len(val_entropy)*0.3)]]
                if 'FANVM' in self.model_name:
                    anchor_val_fea=Trainer.refer_cal_fea(inital_ckp,self.get_dataloader(anchor_val_samples,initial_train))
                    cur_anchor_val_metric=Trainer.test(inital_ckp,self.get_dataloader(anchor_val_samples,initial_train))
                else:
                    anchor_val_fea=Trainer.refer_cal_fea(inital_ckp,self.get_dataloader(anchor_val_samples))
                    cur_anchor_val_metric=Trainer.test(inital_ckp,self.get_dataloader(anchor_val_samples))

                

                unvalid_generated_pool=useful_unvalid_generated_pool+useless_unvalid_generated_pool
                cur_candidate_pool=[]
                for gvid in generated_pool:
                    if gvid not in unvalid_generated_pool:
                        cur_candidate_pool.append(gvid)
                if len(cur_candidate_pool)==0:
                    print('no valid generated pool')
                    break
                dataloader_generation_pool=self.get_dataloader(cur_candidate_pool)

                all_pool_fea=Trainer.refer_cal_fea(inital_ckp,dataloader_generation_pool)

                all_pool_distance={}
                for vid in all_pool_fea.keys():
                    distance=-1
                    for anchor_vid in anchor_val_samples:
                        cur_distance=torch.nn.functional.cosine_similarity(all_pool_fea[vid],anchor_val_fea[anchor_vid],dim=0)
                        if cur_distance>distance:
                            distance=cur_distance
                    all_pool_distance[vid]=distance


                selected_vids=[x[0] for x in sorted(all_pool_distance.items(),key=lambda x:x[1],reverse=True)[:self.al_pool_size]]
                
                train=initial_train+selected_vids
                dataloader_train=self.get_dataloader(train)
                
                
                
                after_ckp=Trainer.train(inital_ckp,dataloader_train,dataloader_val)
                print('After ckp performance on Val:\n')
                after_ckp_test_metric=Trainer.test(after_ckp,dataloader_val)
                print('After ckp performance on Test:\n')
                _=Trainer.test(after_ckp,dataloader_test)
                after_anchor_val_metric=Trainer.test(inital_ckp,self.get_dataloader(anchor_val_samples))
                
                after_ckp_test_f1=after_ckp_test_metric['f1']

                if after_ckp_test_f1>cur_ckp_test_f1:
                    inital_ckp=after_ckp
                    initial_train=train
                    useless_unvalid_generated_pool=[]
                    useful_unvalid_generated_pool=useful_unvalid_generated_pool+selected_vids
                    with open(save_vid_path+self.model_name+"_poolsize"+str(self.al_pool_size)+"_valifold_"+str(fold_i)+"_v2.txt",'a') as f:
                        for vid in selected_vids:
                            f.write(vid+'\n')
                else:
                    initial_train=initial_train
                    inital_ckp=inital_ckp
                    if after_anchor_val_metric['f1']<=cur_anchor_val_metric['f1']:
                        drop_selected_vids=random.sample(selected_vids, k=len(selected_vids)//2)
                        useless_unvalid_generated_pool=useless_unvalid_generated_pool+drop_selected_vids
                print('after test f1:',after_ckp_test_f1)
                print('Current train size:',len(initial_train))
                print('Current valid pool size:',len(generated_pool)-len(unvalid_generated_pool))


            print("============val fold "+str(fold_i)+" =================")
            final_result=Trainer.test(inital_ckp,dataloader_test)
            if fold_i==0:
                fold1_vids_pool=useful_unvalid_generated_pool
            elif fold_i==1:
                fold2_vids_pool=useful_unvalid_generated_pool
            elif fold_i==2:
                fold3_vids_pool=useful_unvalid_generated_pool
            elif fold_i==3:
                fold4_vids_pool=useful_unvalid_generated_pool
            elif fold_i==4:
                fold5_vids_pool=useful_unvalid_generated_pool
                

        
        all_selected_vids_mp={}
        for vid in fold1_vids_pool:
            all_selected_vids_mp[vid]=all_selected_vids_mp.get(vid,0)+1
        for vid in fold2_vids_pool:
            all_selected_vids_mp[vid]=all_selected_vids_mp.get(vid,0)+1
        for vid in fold3_vids_pool:
            all_selected_vids_mp[vid]=all_selected_vids_mp.get(vid,0)+1
        for vid in fold4_vids_pool:
            all_selected_vids_mp[vid]=all_selected_vids_mp.get(vid,0)+1
        for vid in fold5_vids_pool:
            all_selected_vids_mp[vid]=all_selected_vids_mp.get(vid,0)+1

        final_selected_vids=[]
        for vid in all_selected_vids_mp.keys():
            if all_selected_vids_mp[vid]>=3:
                final_selected_vids.append(vid)

        with open(save_vid_path+self.model_name+"_poolsize"+str(self.al_pool_size)+"_valifold_major_vote_selected.txt",'w') as f:
            for vid in final_selected_vids:
                f.write(vid+'\n')

        final_train_set_vids=ori_train_vids+final_selected_vids
        inital_ckp=torch.load(self.init_ckp_path) if self.init_ckp_path is not None else self.model.state_dict()
        Trainer=Trainer_AL(model=self.model,device=self.device,lr=self.lr,dropout=self.dropout,weight_decay=self.weight_decay,model_name=self.model_name,epoch_stop=self.epoch_stop,save_param_path=self.save_param_dir+self.dataset_name+"/AL_temporal/"+self.model_name+"/"+"poolsize"+str(self.al_pool_size)+"/",epoches=self.epoches)
        dataloader_train=self.get_dataloader(final_train_set_vids)

        dataloader_val=self.get_dataloader(or_val_vids)
        dataloader_test=self.get_dataloader(test)

        after_ckp=Trainer.train(inital_ckp,dataloader_train,dataloader_val)
        print('=================Final result on Major Vote===============')
        final_result=Trainer.test(after_ckp,dataloader_test)
        print(final_result)
