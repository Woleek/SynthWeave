import os
import warnings
import numpy as np
import torch
import torch.nn as nn

# ====================================
#             VISUAL BRANCH
# ====================================
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class BasicBlockIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + \
           [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]

class Bottleneck:
    def __init__(self, in_channel, depth, stride):
        self.in_channel = in_channel
        self.depth = depth
        self.stride = stride

def get_blocks(num_layers=50):
    return [
        get_block(64, 64, 3),
        get_block(64, 128, 4),
        get_block(128, 256, 14),
        get_block(256, 512, 3),
    ]

def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None:
                m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()

class Backbone(nn.Module):
    def __init__(self, input_size=(112,112), num_layers=50, mode='ir'):
        super(Backbone, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        blocks = get_blocks(num_layers)
        modules = [BasicBlockIR(b.in_channel, b.depth, b.stride) for block in blocks for b in block]
        self.body = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False)
        )

        initialize_weights(self.modules())

    def forward(self, x):
        x = self.input_layer(x)
        for module in self.body:
            x = module(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)
        return output, norm

def IR_50(input_size=(112, 112)):
    return Backbone(input_size, 50, 'ir')

def load_pretrained_model(path):
    # load model and pretrained statedict
    model = IR_50()
    statedict = torch.load(os.path.join(path, "adaface_ir50_ms1mv2.ckpt"), weights_only=False)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

class AdaFace(nn.Module):
    def __init__(self, path: str, freeze=True):
        super(AdaFace, self).__init__()
        self._prepare_model(path, freeze)

    def _prepare_model(self, path, freeze):
        self.model = load_pretrained_model(path)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, images):
        """
        Expects images as torch.Tensor with shape [B, 3, H, W] and pixel values in [0, 1]
        """
        if not torch.is_tensor(images):
            raise ValueError("Input must be a PyTorch tensor.")

        # Normalize to match AdaFace training: mean=0.5, std=0.5
        images = (images - 0.5) / 0.5

        embeddings, _ = self.model(images)
        return embeddings
    
    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (np.linalg.norm(e_i) * np.linalg.norm(e_j)) * 100
    
class ReDimNet(nn.Module):
    def __init__(self, freeze=True):
        super(ReDimNet, self).__init__()
        self._prepare_model(freeze)
    
    def _prepare_model(self, freeze):
        self.model = torch.hub.load(
            repo_or_dir='IDRnD/ReDimNet', 
            model='ReDimNet',
            model_name='b6',
            train_type='ptn',
            dataset='vox2',
            # force_reload=True
        )
            
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, audios):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            
            # embeddings = []
            
            # for audio in audios:
            #     emb = self.model(audio)
            #     embeddings.append(emb.flatten())
            
            # embeddings = torch.stack(embeddings, dim=0)
            embeddings = self.model(audios)
            return embeddings
    
    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (np.linalg.norm(e_i) * np.linalg.norm(e_j)) * 100