import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import h5py


def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def pad_sequence(seq_len,lst, emb):
    result=[]
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len=video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len,emb],dtype=torch.long)
        elif ori_len>=seq_len:
            if emb == 200:
                video=torch.FloatTensor(video[:seq_len])
            else:
                video=torch.LongTensor(video[:seq_len])
        else:
            video=torch.cat([video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.long)],dim=0)
            if emb == 200:
                video=torch.FloatTensor(video)
            else:
                video=torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)



class SVFENDDataset_tvva_gen(Dataset):

    def __init__(self, input_vids,dataset_name='fakett'):

        if dataset_name=='fakett':
        
            with open('data/fakett/dict_vid_audioconvfea.pkl', "rb") as fr:
                self.dict_vid_convfea = pickle.load(fr)

            self.data_complete = pd.read_json('./data/fakett/enhanced_v1_fakett.json',orient='records',lines=True,dtype={"video_id":str})
            self.text_fea_path='./data/fakett/bert_text_fea.pkl'
            with open(self.text_fea_path, 'rb') as f:
                self.text_fea = torch.load(f)
            self.framefeapath='./data/fakett/ptvgg19_frames'
            self.c3dfeapath='./data/fakett/c3d/'

            self.vids = input_vids
            self.data = self.data_complete[self.data_complete.video_id.isin(self.vids)]  

        elif dataset_name=='fakesv':
            with open('./data/fakesv/dict_vid_audioconvfea.pkl', "rb") as fr:
                self.dict_vid_convfea = pickle.load(fr)

            self.data_complete = pd.read_json('./data/fakesv/enhanced_v1_fakesv.json',orient='records',dtype=False,lines=True)
            self.text_fea_path='./data/fakesv_allgen/bert_text_fea.pkl'
            with open(self.text_fea_path, 'rb') as f:
                self.text_fea = torch.load(f)
            self.framefeapath='./data/fakesv/ptvgg19_frames'
            self.c3dfeapath='./data/fakesv/c3d/'

            self.vid = input_vids
            self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]  

        self.data.reset_index(inplace=True)  
        self.dataset_name=dataset_name

        
    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label 
        if self.dataset_name=='fakett':
            label = 0 if item['label']=='real' else 1
        elif self.dataset_name=='fakesv':
            label = 0 if item['label']=='真' else 1
        label = torch.tensor(label)

        text_fea=self.text_fea['last_hidden_state'][vid]
        
        audio_vid=item['visual_materials'][0].split('_')[0] if '_' in item['visual_materials'][0] else item['visual_materials'][0]
        # # audio
        audioframes = self.dict_vid_convfea[audio_vid]
        audioframes = torch.FloatTensor(audioframes)
        
        # frames
        frames=[]
        for vm in item['visual_materials']:
            vmvid=vm.split('_')[0]
            sframe=int(vm.split('_')[1][1:])
            eframe=int(vm.split('_')[2][1:])
            frms=pickle.load(open(os.path.join(self.framefeapath,vmvid+'.pkl'),'rb'))
            sframe=min(frms.shape[0],sframe)
            eframe=min(frms.shape[0],eframe)
            frames.append(frms[sframe:eframe+1])
        frames=np.concatenate(frames,axis=0)
        # frames=pickle.load(open(os.path.join(self.framefeapath,vid+'.pkl'),'rb'))
        frames=torch.FloatTensor(frames)
        
        # # video
        c3d_id = vid.split('_')[0] if 'TMVO' in vid else vid
        c3d = h5py.File(self.c3dfeapath+c3d_id+".hdf5", "r")[c3d_id]['c3d_features']
        c3d = torch.FloatTensor(c3d)

        return {
            'vid':vid,
            'label': label,
            'text_fea':text_fea,
            'audioframes': audioframes,
            'frames':frames,
            'c3d': c3d
        }

def SVFEND_tvva_gen_collate_fn(batch): 
    num_frames = 83
    num_audioframes = 50 
    vid = [item['vid'] for item in batch]

    text_fea = torch.stack([item['text_fea'] for item in batch])

    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)

    audioframes  = [item['audioframes'] for item in batch]
    audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

    c3d  = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    label = [item['label'] for item in batch]

    return {
        'vid':vid,
        'label': torch.stack(label),
        'text_fea':text_fea,
        'audioframes': audioframes,
        'frames':frames,
        'c3d': c3d,
    }




