from collections import OrderedDict
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import load_model
from mobilenetv2 import MobileNetV2

SIGNAL_LENGTH = 2500 # samples

def merge_all_models():
    project_dir = load_project_config()['project_dir']
    models_dir = os.path.join(project_dir, 'models')
    model_folders = glob.glob('../models/*')
    
    for model_folder in model_folders:
        model_name = Path(model_folder).parent.stem
        merged_outpath = os.path.join(model_folder, 'merged', f'{model_name} merged.pt')
        if os.path.exists(merged_outpath): # delete existing merged files
            os.remove(merged_outpath)
        if model_name == 'Hsieh':
            model_class = Hsieh2020
        elif model_name == 'MobileNetV2':
            model_class = MobileNetV2
        
        args_path = os.path.join(model_folder, 'model_args.json')
        with open(args_path, 'r') as args_file:
            model_args = json.load(args_file)
        model_pattern = os.path.join(model_folder, '*.pt')
        merge_models(model_class, merged_outpath, *glob.glob(model_pattern), **model_args)

def merge_models(model_class, outpath=None, device=torch.device('cpu'), *model_paths, **model_args):
    if outpath is None:
        outfolder = Path(model_paths[0]).parent
        outpath = os.path.join(outfolder, 'merged', f'{model_type} merged.pt')
    
    models = [load_model(model_path, model_class, model_args, device) for model_path in model_paths]
    return MergedModel(*models)

class MergedModel(nn.Module):
    def __init__(
        self,
        *models
    ):
        super().__init__()
        self._modules = OrderedDict( # load models into modules
            **{str(idx): module for idx, module in enumerate(models)}
        )
    
    def forward(self, x):
        out = (
            torch.stack(
                [model(x) for model in self._modules.values()]
            )
        )
        out = torch.exp(out)
        out = torch.mean(out, dim=0)
        return out
    
    def to(self, device):
        for key in self._modules.keys():
            self._modules[key] = self._modules[key].to(device)
        
    def eval(self):
        for key in self._modules.keys():
            self._modules[key].eval()
        
    def train(self): # Don't use this, please
        for key in self._modules.keys():
            print(self._modules[key])
            self._modules[key].train()
    
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ReLU=True,
        max_pool=True,
        batch_norm=False,
        dropout=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        block_struct = [nn.Conv1d(in_channels, out_channels, 5, 1)]
        
        if batch_norm:
            block_struct.append(nn.BatchNorm1d(out_channels))
        if ReLU:
            block_struct.append(nn.ReLU(inplace=True))
        if max_pool:
            block_struct.append(nn.MaxPool1d(2, 2))
        if dropout:
            block_struct.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*block_struct)
    
    def forward(self, x):
        return self.block(x)
        
class Hsieh2020(nn.Module):
    def __init__(
        self, 
        channels,
    ):
        super().__init__()
        self.acc = None
        self.input_size = SIGNAL_LENGTH
        self.input_channels = channels
        
        self.conv_layer = nn.Sequential(
            ConvBlock(2, 32, batch_norm=True),
            ConvBlock(32, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, dropout=0.5),
            ConvBlock(256, 512),
            ConvBlock(512, 512, dropout=0.5),
            nn.Flatten()
        )
        
        self.fc1_size = self.get_fc1_size()
        self.dense_layer = nn.Sequential(
            nn.Linear(self.fc1_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.ReLU(inplace=True),
            nn.LogSoftmax(dim=1)
        )
        
    def get_fc1_size(self):
        self.eval()
        x = torch.zeros(1, self.input_channels, self.input_size)
        x = self.conv_layer(x)
        return x.size()[-1]
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.dense_layer(x)
        return x
        
class Afib_CNN(nn.Module):
    def __init__(
        self, 
        channels,
        repeat_layers=2,
    ):
        super().__init__()
        self.acc = None
        self.input_size = SIGNAL_LENGTH
        self.input_channels = channels
        self.has_repeat_layers = repeat_layers > 0
        
        self.conv_layer = nn.Sequential(
#              ConvBlock(self.input_channels, 1),
            ConvBlock(self.input_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 512),
            nn.Conv1d(512, 512, 1, 1),
            nn.MaxPool1d(2,2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.consecutive_conv = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128, batch_norm=True, dropout=0.5),
        )
        
        if self.has_repeat_layers:
            self.repeat_layers = nn.Sequential(
                *[self.consecutive_conv for i in range(repeat_layers)]
            )
        else:
            self.repeat_layers = None
            
        
        self.reduce = nn.Sequential(
#             ConvBlock(32, 16),
#             ConvBlock(16, 1, batch_norm=True, dropout=0.25),
            nn.Flatten()
        )
        
        self.fc1_size = self.get_fc1_size()
        
        self.linear_layers = nn.Sequential(
            nn.Linear(self.fc1_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
            nn.ReLU(inplace=True),
            nn.LogSoftmax(dim=1)
        )
        
    def get_fc1_size(self):
        x = torch.zeros(3, self.input_channels, self.input_size)
        x = self.conv_layer(x)
        if self.has_repeat_layers: 
            x = self.repeat_layers(x)
        x = self.reduce(x)
        return x.size()[-1]
    
    def forward(self, x):
        x = self.conv_layer(x)
        if self.has_repeat_layers: 
            x = self.repeat_layers(x)
        x = self.reduce(x)
        x = self.linear_layers(x)
        return x