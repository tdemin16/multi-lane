from collections import OrderedDict

from timm.models.layers import Mlp, DropPath

import torch
from torch import nn
from torch.nn import functional as F

from multi_lane.utils import bipartite_soft_matching, merge_wavg


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., id: int=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.token_scale = dim ** -0.5
        self.dim = dim
        self.id = id

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.loss = 0
    
    def get_prompts(self, cls):
        assert hasattr(self, 'prompts')
        T, B, C = cls.shape # tuberculosis
        if self.training:
            prompts = self.prompts[:, self.t:self.t+1]
        else:
            prompts = self.prompts[:, :self.t+1]
        
        # [2, T, K, H, D] -> [2, T, B, H, K, D]
        prompts = prompts.unsqueeze(2).expand(-1, -1, B, -1, -1, -1)
        prompts = prompts.permute(0, 1, 2, 4, 3, 5)
        return prompts

    def forward(self, x, tokens):
        # Batch, Num. patches, Embedding dim
        B, N, C = x.shape

        #? ----------- Patch extraction -----------
        task_x = tokens
        if (not self.disable_dandr or self.id == 0) and self.tome == 0:
            # seprate task class token from tokens
            task_cls = tokens[:, :, 0:1]
            selectors = tokens[:, :, 1:]

            g_selectors = selectors
                
            # compute softmax similarity between global selectors and patches
            T, B, G, C = g_selectors.shape
            # [T, B, S, C] X [B, N, C] -> [T, B, S, N]
            g_patch_attn = torch.einsum('tbsc, bnc -> tbsn', g_selectors, x.detach()) * self.token_scale
            g_soft_patches = g_patch_attn.softmax(dim=-1)

            # aggregate patches according to softmax similarity
            # [B, N, C] X [T, B, S, N, 1] -> [T, B, S, C]
            g_task_patches = torch.einsum('bnc, tbsn -> tbsc', x.detach(), g_soft_patches)

            # compute softmax similarity between local selectors and patches
            task_patches = g_task_patches

            task_x = torch.cat((task_cls, task_patches), dim=2)
        
        elif self.tome > 0: #* Token Merging
            task_x = x.detach().clone()
            while task_x.size(-2) > self.tome: 
                merge, _ = bipartite_soft_matching(
                    task_x, 
                    r=task_x.size(-2) // 2, 
                    class_token=True,
                )
                task_x, _ = merge_wavg(merge, task_x)
            
            task_x = task_x.unsqueeze(0).repeat(tokens.size(0), 1, 1, 1)
        #? ----------------------------------------

        #* ----------- Compute token attention -----------
        T, B, H, C = task_x.shape
        task_qkv = self.qkv(task_x).reshape(T, B, H, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        task_q, task_k, task_v = task_qkv.unbind(0)
        
        if hasattr(self, 'prompts'):
            query_cls = task_x[:, :, 0].clone()
            if self.detach:
                query_cls = query_cls.detach()
            prompts = self.get_prompts(query_cls)
            
            task_k = torch.cat((prompts[0], task_k), dim=-2)
            task_v = torch.cat((prompts[1], task_v), dim=-2)

        task_attn = (task_q @ task_k.transpose(-2, -1)) * self.scale
        task_attn = task_attn.softmax(dim=-1)
        task_attn = self.attn_drop(task_attn)

        task_x = (task_attn @ task_v).transpose(2, 3).reshape(T, B, H, C)
        task_x = self.proj(task_x)
        task_x = self.proj_drop(task_x)
        #* -----------------------------------

        # remove extracted patches from tokens
        if not self.disable_dandr and self.tome == 0:
            fix_part = task_x[:, :, 0:1]
            tokens = torch.cat((fix_part, selectors), dim=2)
        elif self.tome > 0:
            fix_part = task_x[:, :, 0:1]
            tokens = fix_part
        else:
            tokens = task_x

        #! ----------- Compute frozen pass -----------
        with torch.no_grad():
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # [B, N, C] -> [B, N, C*3] -> [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        #! ------------------------------------------
        
        return x, tokens
    
    def init(self, args, use_prompts: bool):
        # curr task
        self.t = 0
        self.num_selectors = args.num_selectors
        self.detach = args.detach
        self.disable_dandr = args.disable_dandr
        self.tome = args.tome
        
        if use_prompts:
            # prompt initialization
            prompts = torch.randn((2, args.num_tasks, args.num_prompts, 
                                   self.num_heads, self.dim // self.num_heads))
            self.prompts = nn.Parameter(prompts)
            if args.prompt_init == 'orthogonal':
                nn.init.orthogonal_(self.prompts)
            elif args.prompt_init == 'uniform':
                nn.init.uniform_(self.prompts, -1, 1)

    def next_task(self):
        self.t += 1
        with torch.no_grad():
            if hasattr(self, 'prompts'):
                if self.prompts.grad is not None:
                    self.prompts.grad.zero_()
                self.prompts[:, self.t] = self.prompts[:, self.t - 1].clone().detach()

            if hasattr(self, 'keys'):
                if self.keys.grad is not None:
                    self.keys.grad.zero_()
                self.keys[self.t] = self.keys[self.t - 1].clone().detach()

                if self.prompts.grad is not None:
                    self.prompts.grad.zero_()
                curr = self.t
                next_ = (self.t + 1)
                prev = (self.t - 1)
                self.prompts[:, curr:next_] = self.prompts[:, prev:curr].clone().detach()


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pret_attention=False, prompts=False,
            id: int=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not pret_attention:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = PreT_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, id=id)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.prompts = prompts
        self.id = id

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if not type(self.attn) == PreT_Attention:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x
        else:
            assert 'task_tokens' in kwargs.keys(), 'tokens must be provided when prompts is True'
            tokens = kwargs['task_tokens']
            x_inter, tokens_inter = self.attn(self.norm1(x), self.norm1(tokens))
            
            x = x + self.drop_path1(self.ls1(x_inter))
            tokens = tokens + self.drop_path1(self.ls1(tokens_inter))
            
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            tokens = tokens + self.drop_path2(self.ls2(self.mlp(self.norm2(tokens))))
            return x, tokens
        
    def get_ortho_loss(self):
        assert self.prompts
        return self.attn.get_ortho_loss()
    
    def init(self, args):
        assert type(self.attn) == PreT_Attention
        self.attn.init(args, self.prompts)

    def next_task(self):
        assert type(self.attn) == PreT_Attention
        self.attn.next_task()
    

class ResPostBlock(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelBlock(nn.Module):

    def __init__(
            self, dim, num_heads, num_parallel=2, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x):
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)
    

class TaskIdentifier(nn.Module):
    def __init__(self, num_tasks: int, embed_dim: int):
        super().__init__()
        self.t = 0
        self.keys = nn.Parameter(torch.randn((num_tasks, embed_dim)))
        nn.init.orthogonal_(self.keys)

    def forward(self, feats: torch.Tensor):
        if self.training:
            key = F.normalize(self.keys[self.t], dim=-1).unsqueeze(0)
            feats = F.normalize(feats, dim=-1)
            sim = torch.sum(feats * key) / feats.size(0)
            tasks = torch.zeros((feats.size(0),), dtype=torch.long)
        else:
            keys = F.normalize(self.keys[:self.t+1], dim=-1)
            feats = F.normalize(feats, dim=-1)
            sim = feats @ keys.T
            sim, tasks = torch.max(sim, dim=-1)
            sim = torch.mean(sim)

        return tasks, sim

    def next_task(self):
        self.t += 1

