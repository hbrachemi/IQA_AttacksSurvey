import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768,img_w: int = 224,img_h: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.proj = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1,(img_w // patch_size) * (img_h // patch_size) + 1, emb_size))
        #print(f"cls token {self.cls_token.shape}   pos token {self.pos_embed.shape}")
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.proj(x)
        
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
    
        
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)

        if x.size(1) != self.pos_embed.size(0):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            


            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            

            P = x.size(1)
            
            
            new_pos_embed = F.interpolate(other_pos_embed, size=(P-1), mode='nearest')
            
                        
            new_pos_embed = new_pos_embed.transpose(1, 2)    

            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)

            x += new_pos_embed
        else:
            x += self.pos_embed
        return x    


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
        
        
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
        
        
        
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
        
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
            
            
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
                
                
class RegressionHead(nn.Sequential):
    def __init__(self, emb_size: int = 768):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            #nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 1),
            nn.ReLU()
            )
            
     
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_w: int = 224,
                img_h: int = 224,
                depth: int = 12,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_w,img_h),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            RegressionHead(emb_size)
        )

