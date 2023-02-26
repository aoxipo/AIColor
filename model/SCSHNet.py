import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from .util import *
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
            Residual(64, oup_dim, self.conv_type_dict[self.Conv_method]),
            Pool(2, 2),
            #Residual(64, 128, self.conv_type_dict[self.Conv_method]),
            Residual(oup_dim,oup_dim, self.conv_type_dict[self.Conv_method]),
        )
        self.break_up = Residual(oup_dim, inp_dim, self.conv_type_dict[self.Conv_method])
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
            nn.Conv2d(nstack*oup_dim,9,2,2),
            nn.Conv2d(9,9,2,2)
        )
        self.head =  nn.Conv2d(9,3,1,1)
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()
        self.up = nn.Upsample(scale_factor=4, mode='nearest')
        
    def forward(self, imgs):
        #print(imgs.size())
        P,C,W,H = imgs.size()

        if( C == 1 or C == 3):
            x = imgs
        else:
            x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
            
        x_backup = x
        x_origin = x 
        x = self.pre(x)
        x = self.break_up(x)
        #print('res:',x.size())
        combined_hm_preds = []
       
        for i in range(self.nstack):
            hg = self.hgs[i](x_backup)
            #print("hg:",hg.size())
            feature = self.features[i](hg)
            #print("feature:",feature.size())
            preds = self.outs[i](feature)
            #print("preds:", preds.size())
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x_backup = x_backup + self.merge_preds[i](preds) + self.merge_features[i](feature)
        multi_map = torch.cat(combined_hm_preds, 1)
        attention_map = torch.mul(self.merge(multi_map),x)
        color_offset = self.head(attention_map)
        x_color = x_origin + self.up(color_offset)
        return x_color#, torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss

class ResUnetBlock(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self,nstack, inp_dim, oup_dim, bn, increase, conv_api):
        super(ResUnetBlock, self).__init__()
        self.hgs = nn.ModuleList( [
            nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        self.features = nn.ModuleList( [
            nn.Sequential(
            Residual(inp_dim, inp_dim, conv_api),
            conv_api(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        self.outs = nn.ModuleList( [conv_api(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.nstack = nstack
    def forward(self, x):
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            #print("hg:",hg.size())
            feature = self.features[i](hg)
            #print("feature:",feature.size())
            preds = self.outs[i](feature)
            #print("preds:", preds.size())
            combined_hm_preds.append(preds)
        return combined_hm_preds 

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
 
class RESUNet_D(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, Conv_method = "Conv",image_shape = (1,256,256),bn=False, increase=0, **kwargs):
        super(RESUNet, self).__init__()
        self.Conv_method = Conv_method
        self.image_shape = image_shape
        self.nstack = nstack
        self.inp_dim = inp_dim 
        self.oup_dim = oup_dim
        self.conv_type_dict = {
            "DWConv":DWConv,
            "Conv":Conv,
        }
        print("using :",Conv_method)
        self.pre = nn.Sequential(
            self.conv_type_dict[self.Conv_method](image_shape[0], 64, 7, 2, bn=True, relu=True),
            Residual(64, 128, self.conv_type_dict[self.Conv_method]),
            Pool(2, 2),
            #Residual(64, 128, self.conv_type_dict[self.Conv_method]),
            Residual(128,128, self.conv_type_dict[self.Conv_method]),
        )
        self.break_up = Residual(128, 1, self.conv_type_dict[self.Conv_method])
        self.hgs_r_pip = ResUnetBlock(nstack, 1, oup_dim, bn, increase, self.conv_type_dict[self.Conv_method])
        self.hgs_g_pip = ResUnetBlock(nstack, 1, oup_dim, bn, increase, self.conv_type_dict[self.Conv_method])
        self.hgs_b_pip = ResUnetBlock(nstack, 1, oup_dim, bn, increase, self.conv_type_dict[self.Conv_method])
        
        #self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        #self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        
        self.merge = nn.ModuleList( [
            nn.Sequential(
            nn.Conv2d(nstack*128,9,2,2),
            nn.Conv2d(9,9,2,2)
        ) for i in range(3)] )
        
        self.head = nn.ModuleList( 
            [nn.Sequential( nn.Conv2d(9,1,1,1) ) for i in range(3)]
        )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()
        self.up = nn.Upsample(scale_factor=4, mode='bicubic')
        
    def forward(self, imgs):
        ## our posenet
        P,C,W,H = imgs.size()

        if( C == 1 or C == 3):
            x = imgs
        else:
            x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
            
        x_backup = x
        x = self.pre(x)
        x = self.break_up(x)
        #print('res:',x.size())

        r = self.hgs_r_pip(x_backup[:,0,:,:].unsqueeze(1))
        g = self.hgs_g_pip(x_backup[:,1,:,:].unsqueeze(1))
        b = self.hgs_b_pip(x_backup[:,2,:,:].unsqueeze(1))
        
        r_multi_map = torch.cat(r, 1)
        g_multi_map = torch.cat(g, 1)
        b_multi_map = torch.cat(b, 1)
        
        
        r_multi_map = self.merge[0](r_multi_map)
        g_multi_map = self.merge[1](g_multi_map)
        b_multi_map = self.merge[2](b_multi_map)
        
        r_attention_map = torch.mul(r_multi_map,x)
        g_attention_map = torch.mul(g_multi_map,x)
        b_attention_map = torch.mul(b_multi_map,x)
        #print(r_attention_map.size())
        color_r_offset = self.head[0](r_attention_map)
        color_g_offset = self.head[1](g_attention_map)
        color_b_offset = self.head[2](b_attention_map)
        color_offset = torch.cat([color_r_offset, color_g_offset, color_b_offset],1)
        x_color = self.up(color_offset)
        #print(x_color.size())
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