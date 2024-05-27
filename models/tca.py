from timm.models.layers import to_2tuple
import torch
from einops import rearrange 
from torch import nn 

class MaskedSliceChannelAttention(nn.Module):

    def __init__(self, dim, slices=12,num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.proj = nn.Conv2d(dim, dim,1,1,0,groups=slices)
   
    def forward(self, q,k,v,H,W,mask=None):
        B, N, C = q.shape
        q = q.view(B,N,C//self.num_heads,self.num_heads).permute(0,3,2,1).contiguous() 
        k = k.view(B,N,C//self.num_heads,self.num_heads).permute(0,3,2,1).contiguous() 
        v = v.view(B,N,C//self.num_heads,self.num_heads).permute(0,3,2,1).contiguous()   # B H C//H N
 

        q = q * self.scale
        attention = (q @ k.transpose(-2, -1))
        if mask is not None:
            attention = attention.masked_fill_(mask, float("-inf"))

        attention = attention.softmax(dim=-1) # B H C//H C//H

        x = (attention @ v) # B H C//H N
        x = x.permute(0,3,2,1).reshape(B,N,C).permute(0,2,1).contiguous() # B H C//H N -> B N C//H H
   
        x = x.view(B,C,H,W)
        x = self.proj(x)
        return x
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            slices=12,
            act_layer=nn.GELU()):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,1,1,0,groups=slices)
        self.act = act_layer()
        self.fc2 = nn.Conv2d( hidden_features,in_features,1,1,0,groups=slices)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, slices=12,act=False, normtype=False):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)
        self.norm = nn.GroupNorm(slices, dim)
    
        self.activation = nn.GELU()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.norm(feat)
        x = x + self.activation(feat)
        return x
    
class TCA_Block(nn.Module):

    def __init__(self, dim, num_heads=16, slices=12,mlp_ratio=1., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()
        self.q1 = nn.Conv2d(dim,dim,1,1,0,groups=slices)
        self.k1 = nn.Conv2d(dim,dim,1,1,0,groups=slices)
        self.v1 = nn.Conv2d(dim,dim,1,1,0,groups=slices)
        self.num_heads = num_heads
        self.norm1 = nn.GroupNorm(slices, dim)
        self.norm2 = nn.GroupNorm(slices, dim)
        self.slices = slices
        self.window_size=8
       
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim,slices=slices, k=3, act=cpe_act)
                                 ])
        self.ffn = ffn
   
        self.attn = MaskedSliceChannelAttention(dim, slices=slices, num_heads=num_heads, qkv_bias=qkv_bias)
    

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            slices=slices,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer)
        self.mask = self.generate_mask(dim)
        
       
    def generate_mask(self,dim):
        heads_dim = dim //self.num_heads
        attn_mask = torch.zeros(1, heads_dim, heads_dim, dtype=torch.bool)
        for i in range(self.slices-1):
            attn_mask[:,:(i+1)*heads_dim//self.slices,(i+1)*heads_dim//self.slices:(i+2)*heads_dim//self.slices] = 1
        return attn_mask
    def forward(self, y):
 
        tgt = y

        y = rearrange(y, 'b c (w1 p1) (w2 p2)  -> b c w1 w2 p1 p2 ', p1=self.window_size, p2=self.window_size)
        
        B,C,W1,W2,H,W = y.shape
       
        y = rearrange(y, 'b c w1 w2 p1 p2  -> (b w1 w2) c p1 p2 ', p1=self.window_size, p2=self.window_size)
        
        y = self.norm1(y)
        y = self.cpe[0](y)
       

        q =self.q1(y).flatten(2).permute(0,2,1)
        k =self.k1(y).flatten(2).permute(0,2,1)
        v =self.v1(y).flatten(2).permute(0,2,1)   # B N C
        attn_result= self.attn(q,k,v,H,W,self.mask.to(q.device))
        
        attn_result = rearrange( attn_result, '(b w1 w2) c p1 p2  -> b c (w1 p1) (w2 p2)',w1=W1,w2=W2)
    
        tgt = tgt + attn_result

        tgt= tgt + self.mlp(self.norm2(tgt))
        return tgt




class TCA(nn.Module):
    def __init__(self, dim=192,depth=4,ratio=4,slices=12,drop_path_rate=0.3):
        super().__init__()
        self.num_layers = depth
        self.slices= slices
        self.start_token = nn.Parameter(torch.zeros(1,dim//self.slices,1,1))
        self.lift = nn.Conv2d(dim, dim*ratio, 3,1,1,groups=self.slices)
        self.start_token_from_hyperprior= nn.Conv2d(dim*2, dim//self.slices, 3,1,1)
        self.dim = dim 
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer =TCA_Block(
                dim=dim*ratio,slices=slices)
            self.layers.append(layer)
  
    def forward(self,hyper,y):
        B,C,H,W = y.shape
        start_token = self.start_token_from_hyperprior(hyper)
        out = self.lift(torch.cat((start_token,y[:,:-C//self.slices]),dim=1))

        for i in range(self.num_layers):
            out= self.layers[i](out)

        return out



class TCA_EntropyModel(nn.Module):
    def __init__(self, dim=192,depth=4,ratio=4,slices=12):
        super().__init__()
        self.slices = slices

        self.ratio =ratio
        self.TCA= TCA(dim=dim,slices=self.slices,depth=depth,ratio=self.ratio) # B rC H W   
        self.hyper_trans = nn.Linear(dim*2,dim*2)  # B 2C H W 
        self.entropy_parameters_net= nn.Sequential(
            nn.Conv2d(dim*(self.ratio+2), dim*self.ratio//2, 3,1,1,groups=self.slices),
            nn.GELU(),
            nn.Conv2d(dim*self.ratio//2, dim*3, 3,1,1,groups=self.slices),
            nn.GELU(),
            nn.Conv2d(dim*3, dim*3, 3,1,1,groups=self.slices),
           
        )

       
    def forward(self,hyper,y):
        B,C,H,W = y.shape

        hyper1 = self.hyper_trans(hyper.flatten(2).permute(0,2,1)).permute(0,2,1).view(B,C,-1,H,W)

        out1 = self.TCA(hyper,y).view(B,C,-1,H,W).contiguous()
      
        out = self.entropy_parameters_net(torch.cat((out1,hyper1),2).view(B,C*(self.ratio+2),H,W)).view(B,C,3,H,W).contiguous() # B 2C H W
       
       
        means = out[:,:,0].contiguous()
        scales = out[:,:,1].contiguous()
        lrp = out[:,:,2].contiguous()
 
        return means,scales,lrp

if __name__ == "__main__":
    slices=10
    casual_test = TCA_EntropyModel(dim=320,slices=10).cuda()
    y = torch.randn(1,320,16,16).cuda()
    hyper = torch.randn(1,640,16,16).cuda()
    means1,scales1,lrp1 = casual_test(hyper,y)
    import numpy as np


    # y2 = y
    for i in range(slices):
        k = i*320//slices
        y2= y.clone()
        y2[0][k:]=0
        means2,scales2,lrp2 = casual_test(hyper,y2)
        print(i)
        print((means1-means2).abs().mean(-1).mean(-1)[0][:k].mean(),(means1-means2).abs().mean(-1).mean(-1)[0][k:].mean())
        print((scales1-scales2).abs().mean(-1).mean(-1)[0][:k].mean(),(scales1 -scales2).abs().mean(-1).mean(-1)[0][k:].mean())
        print((lrp1-lrp2).abs().mean(-1).mean(-1)[0][:k].mean(),(lrp1 -lrp2).abs().mean(-1).mean(-1)[0][k:].mean())
    