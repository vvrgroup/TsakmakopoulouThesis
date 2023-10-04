# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

        self.conv3d_C=nn.Conv3d(2,1,kernel_size=3,stride=1,padding=1)
        #self.conv3d_C=nn.Conv3d(2,1,kernel_size=1,stride=1,padding=0)
        self.relu=nn.ReLU()

        self.conv3d_C_2=nn.Conv3d(1,1,kernel_size=3,stride=1,padding=1)

        
    def forward(self, x, record_len,fusion_method):
      out=[]
      if fusion_method==1:
        out=self.attentiveFusion(x,record_len)
  
      if fusion_method==2:
        out=self.adaptiveFusionA(x,record_len)

      if fusion_method==3:
        out=self.adaptiveFusionB(x, record_len)

      if fusion_method==4:
        out=self.adaptiveFusionC(x,record_len)

      if fusion_method==5:
        out=self.adaptiveFusionD(x,record_len)
        
      return out

    def regroup(self, x, record_len):
      cum_sum_len = torch.cumsum(record_len, dim=0)
      split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
      return split_x

    #fusion method 1
    def attentiveFusion(self,x,record_len):
      split_x = self.regroup(x, record_len)
      batch_size = len(record_len)
      C, W, H = split_x[0].shape[1:]
      out = []
      for xx in split_x:
        cav_num = xx.shape[0]
        xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
        h = self.att(xx, xx, xx)
        h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
        out.append(h)
      return torch.cat(out, dim=0)
      #return out

    #fusion method 2
    #A) Spatial fusion with Max pooling (MaxFusion)
    def adaptiveFusionA(self,x,record_len):
      split_x = self.regroup(x, record_len)
      batch_size = len(record_len)
      C, W, H = split_x[0].shape[1:]

      out=[]
        
      for xx in split_x:
        cav_num = xx.shape[0]
        xx = torch.max(xx, dim=0, keepdim=True)[0]
        out.append(xx)
      return torch.cat(out, dim=0)
      
    #fusion method 3
    #B) Spatial fusion with Avg pooling (AvgFusion)
    def adaptiveFusionB(self,x,record_len):
      split_x = self.regroup(x, record_len)
      out=[]
      for xx in split_x:
        xx=torch.mean(xx,dim=0, keepdim=True)
        out.append(xx)
      return torch.cat(out,dim=0)

    #fusion method 4
    #C) Spatial-wise Adaptive feature Fusion (S-AdaFusion)
    def adaptiveFusionC(self, x, record_len):
      split_x = self.regroup(x, record_len)
      out=[]
      for xx in split_x:
        xx1=torch.max(xx, dim=0, keepdim=True)[0]
        xx2=torch.mean(xx,dim=0, keepdim=True)
        
        out.append(torch.cat((xx1, xx2), dim=0).unsqueeze(0))
      out= torch.cat(out, dim=0)
      out=self.conv3d_C(out)
      out=self.relu(out).squeeze(1)
      return out

    #fusion method 5
    #D Channel Fusion with 3D convolution (C-3DFusion)
    def adaptiveFusionD(self, x, record_len):
      split_x = self.regroup(x, record_len)
      out=[]
      for xx in split_x:
        cav_num = xx.shape[0]
        c=nn.Conv3d(cav_num,1,kernel_size=1)
        xx=c(xx)

        out.append(xx)
      return torch.cat(out, dim=0)


##fusion after the completion of backbone ------ S-AdaFusion
class SAdaFusion(nn.Module):
    def __init__(self):
        super(SAdaFusion, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(2, 1, 3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )


    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        out = []

        for xx in split_x:
            x_max = torch.max(xx, dim=0, keepdim=True)[0]
            x_mean = torch.mean(xx, dim=0, keepdim=True)
            out.append(torch.cat((x_max, x_mean), dim=0).unsqueeze(0))
        out = torch.cat(out, dim=0)
        out = self.conv3d(out).squeeze(1)
        return out
        

