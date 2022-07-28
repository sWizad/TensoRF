from .tensoRF import *
import torch

class CNNsingle(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super().__init__()
        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        cnn1 = torch.nn.Conv2d(self.in_mlpC, 3, kernel_size = 3, stride=1, padding='same', padding_mode='reflect')
        self.mlp = torch.nn.Sequential(cnn1)


    def forward(self, pts, viewdirs, features):
        """
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        print(mlp_in.shape)
        exit()
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)
        return rgb
        """
        print(features.shape)
        return features[:,:3]

       
class DeepPix2Pix(TensorVMSplit):
    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'CNNsingle':
            self.renderModule = CNNsingle(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)