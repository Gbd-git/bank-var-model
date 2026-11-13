"""Small independent-bank networks, vectorized without parameter sharing.

Tensor convention [sample, bank, ..., feature]. Each parameter's first axis
is bank. LayerNorm statistics use features only, never banks or samples.
"""
import math
import torch
from torch import nn

class BankLinear(nn.Module):
    def __init__(self,banks,ins,outs):
        super().__init__();self.weight=nn.Parameter(torch.randn(banks,ins,outs)/math.sqrt(ins))
        self.bias=nn.Parameter(torch.zeros(banks,outs))
    def forward(self,x):
        if x.ndim==3:return torch.einsum('bni,nio->bno',x,self.weight)+self.bias
        return torch.einsum('bnti,nio->bnto',x,self.weight)+self.bias[:,None,:]

class BankNet(nn.Module):
    def __init__(self,kind,banks,features,hidden=12):
        super().__init__();self.kind=kind;self.banks=banks;self.hidden=hidden
        if kind=='ml_mlp':
            self.first=BankLinear(banks,features,24);self.second=BankLinear(banks,24,hidden)
        elif kind in ('ml_lstm','ml_gru'):
            gates=4 if kind=='ml_lstm' else 3
            self.input=BankLinear(banks,2,hidden*gates);self.recurrent=BankLinear(banks,hidden,hidden*gates)
        elif kind=='ml_tcn':
            self.input=BankLinear(banks,2,hidden)
            self.convs=nn.ParameterList([nn.Parameter(torch.randn(banks,3,hidden,hidden)/math.sqrt(3*hidden)) for _ in range(3)])
            self.biases=nn.ParameterList([nn.Parameter(torch.zeros(banks,hidden)) for _ in range(3)])
        elif kind=='ml_transformer':
            self.input=BankLinear(banks,2,hidden);self.qkv=BankLinear(banks,hidden,hidden*3)
            self.attout=BankLinear(banks,hidden,hidden);self.ff1=BankLinear(banks,hidden,hidden*2);self.ff2=BankLinear(banks,hidden*2,hidden)
            pos=torch.arange(20)[:,None];freq=torch.exp(torch.arange(0,hidden,2)*(-math.log(10000.)/hidden))
            pe=torch.zeros(20,hidden);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
            self.register_buffer('positional',pe)
        else:raise ValueError(kind)
        self.out=BankLinear(banks,hidden,2)
    def forward(self,x,seq):
        if self.kind=='ml_mlp':h=torch.tanh(self.second(torch.tanh(self.first(x))))
        elif self.kind in ('ml_lstm','ml_gru'):
            h=torch.zeros((seq.shape[0],self.banks,self.hidden),dtype=seq.dtype,device=seq.device);c=torch.zeros_like(h)
            inp=self.input(seq)
            for t in range(seq.shape[2]):
                rec=self.recurrent(h)
                if self.kind=='ml_lstm':
                    i,f,g,o=(inp[:,:,t]+rec).chunk(4,-1)
                    c=torch.sigmoid(f+1.)*c+torch.sigmoid(i)*torch.tanh(g);h=torch.sigmoid(o)*torch.tanh(c)
                else:
                    ir,iz,inn=inp[:,:,t].chunk(3,-1);hr,hz,hn=rec.chunk(3,-1)
                    reset=torch.sigmoid(ir+hr);update=torch.sigmoid(iz+hz)
                    new=torch.tanh(inn+reset*hn);h=(1-update)*new+update*h
        elif self.kind=='ml_tcn':
            h=torch.tanh(self.input(seq))
            for dilation,w,bias in zip((1,2,4),self.convs,self.biases):
                z=torch.zeros_like(h)
                for j in range(3):
                    lag=j*dilation;v=h if lag==0 else torch.nn.functional.pad(h[:,:,:-lag],(0,0,lag,0))
                    z=z+torch.einsum('bnti,nio->bnto',v,w[:,j])
                h=torch.tanh(h+z+bias[:,None,:])
            h=h[:,:,-1]
        else:
            h=self.input(seq)+self.positional;heads=3;d=self.hidden//heads
            q,k,v=[a.reshape(*h.shape[:3],heads,d).permute(0,1,3,2,4) for a in self.qkv(h).chunk(3,-1)]
            score=torch.einsum('bnhtd,bnhsd->bnhts',q,k)/math.sqrt(d)
            mask=torch.triu(torch.ones(20,20,dtype=torch.bool,device=h.device),diagonal=1)
            score=score.masked_fill(mask,float('-inf'))
            att=torch.einsum('bnhts,bnhsd->bnhtd',score.softmax(-1),v).permute(0,1,3,2,4).reshape_as(h)
            h=torch.nn.functional.layer_norm(h+self.attout(att),(self.hidden,))
            h=torch.nn.functional.layer_norm(h+self.ff2(torch.relu(self.ff1(h))),(self.hidden,))
            h=h[:,:,-1]
        return self.out(h)
    def clip_grad(self,maxnorm):
        params=[p for p in self.parameters() if p.grad is not None]
        norms=torch.zeros(self.banks)
        for p in params:norms+=p.grad.reshape(self.banks,-1).square().sum(1)
        factor=(maxnorm/(norms.sqrt()+1e-6)).clamp(max=1.)
        for p in params:p.grad.mul_(factor.reshape(self.banks,*([1]*(p.ndim-1))))
