import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *

from .coattention import *
from .layers import *
from utils.metrics import *

class SimpleSVFEND_tvva(torch.nn.Module):
    def __init__(self,fea_dim,dropout):
        super(SimpleSVFEND_tvva, self).__init__()

        self.text_dim = 768
        self.img_dim = 4096
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50

        self.dim = fea_dim
        self.num_heads = 4

        self.dropout = dropout

        self.attention = Attention(dim=self.dim,heads=4,dropout=dropout)

        self.vggish_layer = torch.hub.load('./pretrain_model/torchvggish', 'vggish', source = 'local')        
        net_structure = list(self.vggish_layer.children())      
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])

        self.co_attention_ta = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
                                        visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)
        self.co_attention_tv = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
                                        visual_len=self.num_frames, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)
        self.trm = nn.TransformerEncoderLayer(d_model = self.dim, nhead = 2, batch_first = True)


        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(fea_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))

        self.classifier = nn.Linear(fea_dim,2)
    
    def forward(self,  **kwargs):
        ### Text Frames ###
        fea_text=kwargs['text_fea']
        fea_text=self.linear_text(fea_text) 

        ### Audio Frames ###
        audioframes=kwargs['audioframes']#(batch,36,12288)
        fea_audio = self.vggish_modified(audioframes) #(batch, frames, 128)
        fea_audio = self.linear_audio(fea_audio) 
        fea_audio, fea_text = self.co_attention_ta(v=fea_audio, s=fea_text, v_len=fea_audio.shape[1], s_len=fea_text.shape[1])
        fea_audio = torch.mean(fea_audio, -2)

        ### Image Frames ###
        frames=kwargs['frames']#(batch,30,4096)
        fea_img = self.linear_img(frames) 
        fea_img, fea_text = self.co_attention_tv(v=fea_img, s=fea_text, v_len=fea_img.shape[1], s_len=fea_text.shape[1])
        fea_img = torch.mean(fea_img, -2)

        fea_text = torch.mean(fea_text, -2)

        ### C3D ###
        c3d = kwargs['c3d'] # (batch, 83, 4096)
        fea_video = self.linear_video(c3d) #(batch, frames, 128)
        fea_video = torch.mean(fea_video, -2)


        fea_text = fea_text.unsqueeze(1)
        fea_img = fea_img.unsqueeze(1)
        fea_audio = fea_audio.unsqueeze(1)
        fea_video = fea_video.unsqueeze(1)
        
        fea=torch.cat((fea_text,fea_audio, fea_video,fea_img),1) # (bs, 4, 128)
        fea = self.trm(fea)
        fea = torch.mean(fea, -2)
        
        output = self.classifier(fea)

        return output,fea
 