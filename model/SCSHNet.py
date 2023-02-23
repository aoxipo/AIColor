import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util import *
from torchsummary import summary

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize

class RESUNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, Conv_method = "Conv",bn=False, increase=0, **kwargs):
        super(RESUNet, self).__init__()
        self.Conv_method = Conv_method
        self.nstack = nstack
        self.inp_dim = inp_dim 
        self.oup_dim = oup_dim
        self.conv_type_dict = {
            "DWConv":DWConv,
            "Conv":Conv,
        }
        print("using :",Conv_method)
        self.pre = nn.Sequential(
            self.conv_type_dict[self.Conv_method](inp_dim, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128, self.conv_type_dict[self.Conv_method]),
            Pool(2, 2),
            #Residual(64, 128, self.conv_type_dict[self.Conv_method]),
            Residual(128,128, self.conv_type_dict[self.Conv_method]),
        )
        self.break_up = Residual(128, inp_dim, self.conv_type_dict[self.Conv_method])
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim, self.conv_type_dict[self.Conv_method]),
            self.conv_type_dict[self.Conv_method](inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [self.conv_type_dict[self.Conv_method](inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.merge = nn.Sequential(
            nn.Conv2d(nstack*128,9,2,2),
            nn.Conv2d(9,9,2,2)
        )
        self.head =  nn.Conv2d(9,3,1,1)
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()
        self.up = nn.Upsample(scale_factor=4, mode='nearest')
        
    def forward(self, imgs):
        ## our posenet
        P,C,W,H = imgs.size()

        if( C == 1 or C == 3):
            x = imgs
        else:
            x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
            
        x_backup = x
        x_origin = x 
        x = self.pre(x)
        x = self.break_up(x)
        print('res:',x.size())
        combined_hm_preds = []
       
        for i in range(self.nstack):
            hg = self.hgs[i](x_backup)
            print("hg:",hg.size())
            feature = self.features[i](hg)
            print("feature:",feature.size())
            preds = self.outs[i](feature)
            print("preds:", preds.size())
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x_backup = x_backup + self.merge_preds[i](preds) + self.merge_features[i](feature)
        multi_map = torch.cat(combined_hm_preds, 1)
        attention_map = torch.mul(self.merge(multi_map),x)
        color_offset = self.head(attention_map)
        x_color = x_origin + self.up(color_offset)
        return x_color #torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss

if __name__ == '__main__':
    image = torch.zeros(1,1,224,224)
    model = RESUNet( 1, 1, 128, Conv_method = "Conv",bn=False)
    model(image).shape
    summary(model.cuda(), input_size=(1,224,224))
    '''
    res: torch.Size([1, 1, 56, 56])
    hg: torch.Size([1, 1, 224, 224])
    feature: torch.Size([1, 1, 224, 224])
    preds: torch.Size([1, 128, 224, 224])
    torch.Size([1, 3, 224, 224])
    '''