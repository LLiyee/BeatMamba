import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional, Tuple
import math

from timm.models.layers import DropPath, trunc_normal_, lecun_normal_
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from utils import make_pad_mask

class AudioCNNFrontend(nn.Module):
    """
    支持双通道输入
    用于音频特征的层次化处理
    """
    def __init__(self, dmodel=256, dropout=0.1,fusion_type='concat'):
        super().__init__()
        self.dmodel = dmodel
        self.fusion_type = fusion_type 

        if fusion_type == 'concat':
            in_channels = 2
            self._process_input = self._concat_channels


        self.conv1_mix = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding=(2, 0))
        self.maxpool1_mix = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout1_mix = nn.Dropout(p=dropout)


        self.conv2_mix = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0))
        self.maxpool2_mix = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout2_mix = nn.Dropout(p=dropout)

        self.conv3_mix = nn.Conv2d(in_channels=64, out_channels=dmodel, kernel_size=(2, 6), stride=1, padding=(1, 0))
        self.maxpool3_mix = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout3_mix = nn.Dropout(p=dropout)


        self.conv1_drum = nn.Conv2d(1, out_channels=32, kernel_size=(3, 3), stride=1, padding=(2, 0))
        self.maxpool1_drum = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout1_drum = nn.Dropout(p=dropout)

        self.conv2_drum = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0))
        self.maxpool2_drum = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout2_drum = nn.Dropout(p=dropout)

        self.conv3_drum = nn.Conv2d(in_channels=64, out_channels=dmodel, kernel_size=(2, 6), stride=1, padding=(1, 0))
        self.maxpool3_drum = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout3_drum = nn.Dropout(p=dropout)
    
    def _concat_channels(self,x):
        # x shape:(batch,time,features,2) -> (batch,2,time,features)
        x = x.permute(0,3,1,2)
        return x
    
    def _process_channel(self,x,conv1,maxpool1,dropout1,
                         conv2,maxpool2,dropout2,
                         conv3,maxpool3,dropout3):
        """处理单个通道"""
        x = conv1(x)
        x = x[:,:,:-2,:]
        x = maxpool1(x)
        x = torch.relu(x)
        x = dropout1(x)

        x = conv2(x)
        x = maxpool2(x)
        x = torch.relu(x)
        x = dropout2(x)

        x = conv3(x)
        x = x[:,:,:-1,:]
        x = maxpool3(x)
        x = torch.relu(x)
        x = dropout3(x)

        # 重塑为(batch, time, dmodel)
        x = x.transpose(1, 3).squeeze(-1).contiguous()
        x = x.squeeze(1)
        return x


    def forward(self, x):
        """
        Args:
            x: (batch, time, features, 2) 
        Returns:
            x_mix: (batch, time, dmodel)
            x_drum: (batch, time, dmodel)

        """
        x_mix = x[...,0].unsqueeze(1)
        x_drum = x[...,1].unsqueeze(1)

        x_mix = self._process_channel(x_mix,self.conv1_mix,self.maxpool1_mix,self.dropout1_mix,
                                      self.conv2_mix,self.maxpool2_mix,self.dropout2_mix,
                                      self.conv3_mix,self.maxpool3_mix,self.dropout3_mix)

        x_drum = self._process_channel(x_drum,self.conv1_drum,self.maxpool1_drum,self.dropout1_drum,
                                       self.conv2_drum,self.maxpool2_drum,self.dropout2_drum,
                                       self.conv3_drum,self.maxpool3_drum,self.dropout3_drum)

        return x_mix, x_drum


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class MambaBlock(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=True, 
        residual_in_fp32=True, drop_path=0.
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # #append
        # # 双向Mamba
        # self.mixer_forward = mixer_cls(dim)
        # self.mixer_backward = mixer_cls(dim)
        # self.norm = norm_cls(dim)
        # self.gate = nn.Parameter(torch.ones(1))
        # self.fusion_linear = nn.Linear(dim * 2, dim)
        # #end
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        # # 前向Mamba处理
        # forward_output = self.mixer_forward(hidden_states, inference_params=inference_params)
        
        # # 后向Mamba处理
        # backward_input = torch.flip(hidden_states, dims=[1])  
        # backward_output = self.mixer_backward(backward_input, inference_params=inference_params)
        # backward_output = torch.flip(backward_output, dims=[1]) 
        
        # # 融合特征
        # combined = torch.cat([forward_output, backward_output], dim=-1)
        # output = self.fusion_linear(combined)
        
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


def create_mamba_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    
    mixer_cls = partial(
        Mamba, 
        layer_idx=layer_idx, 
        bimamba_type=bimamba_type,
        # bimamba_type="none",
        **ssm_cfg, 
        **factory_kwargs
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, 
        eps=norm_epsilon, 
        **factory_kwargs
    )
    
    block = MambaBlock(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class MambaBeatTracker(nn.Module):
    
    def __init__(
        self,
        input_features: int = 256,  
        input_channels: int = 2, 
        dmodel: int = 256,       
        depth: int = 6,        
        nhead: int = 2,          
        num_classes: int = 2,    
        ssm_cfg=None,
        drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True, 
        initializer_cfg=None,
        fused_add_norm: bool = True,  
        residual_in_fp32: bool = True,
        device=None,
        dtype=None,
        bimamba_type: str = "v2",
        tempo_classes: int = 300,  
        num_fusion_layers: int = 2, 
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.dmodel = dmodel
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.num_classes = num_classes
        self.tempo_classes = tempo_classes
        self.input_channels = input_channels
        # self.fusion_type = fusion_type
        
        self.cnn_frontend = AudioCNNFrontend(dmodel=dmodel, dropout=drop_rate)#, fusion_type=fusion_type)

        self.fusion_layers = nn.ModuleList([
            ConcatMambaFusionBlock1D(
                hidden_dim = dmodel,
                drop_path = drop_path_rate * i /num_fusion_layers,
                d_state=16,
                ssm_ratio=2.0,
                mlp_ratio=4.0,
                attn_drop_rate=drop_rate,
                use_checkpoint=False,
                **factory_kwargs,
            )
            for i in range(num_fusion_layers)
        ])

        self.pos_encoding = PositionalEncoding(dmodel, dropout=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.layers = nn.ModuleList([
            create_mamba_block(
                dmodel,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                drop_path=dpr[i],
                bimamba_type=bimamba_type,
                **factory_kwargs,
            )
            for i in range(depth)
        ])
        
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            dmodel, eps=norm_epsilon, **factory_kwargs
        )
        
        # Beat预测头
        self.head_dropout = nn.Dropout(p=drop_rate)
        self.beat_head = nn.Linear(dmodel, num_classes)
        
        # Tempo预测
        self.tempo_dropout = nn.Dropout(p=0.5)
        self.tempo_head = nn.Linear(dmodel, tempo_classes)
        
        self.tempo_norm = nn.LayerNorm(dmodel)
        self.apply_init_weights(depth, initializer_cfg)
    
    def apply_init_weights(self, depth, initializer_cfg):
        """初始化权重"""
        self.cnn_frontend.apply(segm_init_weights)
        self.beat_head.apply(segm_init_weights)
        self.tempo_head.apply(segm_init_weights)
        
        # Mamba specific initialization
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    
    def forward_features(self, x, inference_params=None):
        """
        Args:
            x: (batch, time, features, 2)
            
        Returns:
            hidden_states
            tempo_features 
        """
        # CNN前端处理
        x_mix, x_drum = self.cnn_frontend(x)  

        #特征融合
        for fusion_layer in self.fusion_layers:
            x_mix, x_drum = fusion_layer(x_mix, x_drum)  
            x_mix = x_fused
            x_drum = x_fused
        
        # 位置编码
        x = self.pos_encoding(x_fused)
        tempo_features_list = []
        
        # Mamba layers
        residual = None
        hidden_states = x
        
        for i, layer in enumerate(self.layers):
           # print("hidden_states.shape before mamba:",hidden_states.shape)
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            
            if i % 3 == 0:  
                tempo_features_list.append(hidden_states.mean(dim=1))  # (batch, dmodel)
        
        # Final norm
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if tempo_features_list:
            tempo_features = torch.stack(tempo_features_list, dim=-1).mean(dim=-1)  # (batch, dmodel)
        else:
            tempo_features = hidden_states.mean(dim=1)
            
        return hidden_states, tempo_features
    
    def forward(self, x, return_tempo=True, inference_params=None):
        """
        
        Args:
            x:  (batch, time, features, 2)
            return_tempo
            
        Returns:
            beat_output:  (batch, time, num_classes)  
            tempo_output: (batch, tempo_classes) 
        """
        # 特征提取
        hidden_states, tempo_features = self.forward_features(x, inference_params)
        
        # Beat预测：每个时间步
        beat_features = self.head_dropout(hidden_states)  
        beat_output = self.beat_head(beat_features)  

        if return_tempo:
            # Tempo预测：全局特征
            tempo_features = self.tempo_norm(tempo_features)  
            tempo_features = torch.relu(tempo_features)
            tempo_features = self.tempo_dropout(tempo_features)
            tempo_output = self.tempo_head(tempo_features)  
            return beat_output, tempo_output
        
        return beat_output


