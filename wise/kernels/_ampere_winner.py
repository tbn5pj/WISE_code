"""Experimental Ampere forward kernels. Every attention MMA tile is32x32.

Sparse routing decisions are inputs; this module never selects support.
Regular traversal computes extra32x32 tiles with exact pre-softmax masking.
Persistent CTAs process distinct32-token query rows sequentially, not coarsened.
"""
from __future__ import annotations
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _make_bits(COUNT,IDS,BITS,M:tl.constexpr,WORDS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);word=tl.program_id(1)
    n=tl.load(COUNT+row);x=tl.arange(0,BLOCK)
    key=tl.load(IDS+row*M+x,x<n,other=0)
    valid=(x<n)&(key//32==word)
    bits=tl.where(valid,tl.full((BLOCK,),1,tl.uint32)<<(key%32),0)
    # Keys are unique: integer sum equals bitwise OR.
    value=tl.sum(bits,0).to(tl.uint32)
    tl.store(BITS+row*WORDS+word,value)


@triton.jit
def _row(Q,K,V,O,PTR,COL,IDS,BITS,ROW,
         N:tl.constexpr,H:tl.constexpr,M:tl.constexpr,
         QN:tl.constexpr,QH:tl.constexpr,KN:tl.constexpr,KH:tl.constexpr,VN:tl.constexpr,VH:tl.constexpr,
         ON:tl.constexpr,OH:tl.constexpr,
         MODE:tl.constexpr,THRESH:tl.constexpr,STAGES:tl.constexpr,UNROLL:tl.constexpr,
         META:tl.constexpr,PREFETCH:tl.constexpr,RELOAD_Q:tl.constexpr,EXP2:tl.constexpr,SPLIT_D:tl.constexpr):
    qb=ROW%M;head=ROW//M
    qt=qb*32+tl.arange(0,32);kk=tl.arange(0,32)
    begin=tl.load(PTR+ROW);end=tl.load(PTR+ROW+1);count=end-begin
    if MODE==0:regular=False
    elif MODE==1:regular=True
    else:regular=count*100>=THRESH*(qb+1)
    total=qb+1 if regular else count
    if SPLIT_D:
        d0=tl.arange(0,64);d1=64+tl.arange(0,32)
        if not RELOAD_Q:
            q0=tl.load(Q+qt[:,None]*QN+head*QH+d0[None,:],qt[:,None]<N,0)
            q1=tl.load(Q+qt[:,None]*QN+head*QH+d1[None,:],qt[:,None]<N,0)
        acc0=tl.full((32,64),0.,tl.float32);acc1=tl.full((32,32),0.,tl.float32)
    else:
        d=tl.arange(0,128)
        if not RELOAD_Q:q=tl.load(Q+qt[:,None]*QN+head*QH+d[None,:],(qt[:,None]<N)&(d[None,:]<96),0)
        acc=tl.full((32,128),0.,tl.float32)
    m=tl.full((32,),-float('inf'),tl.float32);l=tl.full((32,),0.,tl.float32)
    if PREFETCH and MODE==0:
        if META==0:next_k=tl.load(COL+begin)
        else:next_k=tl.load(IDS+ROW*M)
    for pos in tl.range(0,total,num_stages=STAGES,loop_unroll_factor=UNROLL):
        if regular:
            kb=pos
            bits=tl.load(BITS+ROW*triton.cdiv(M,32)+kb//32)
            selected=((bits>>(kb%32))&1)!=0
        else:
            if PREFETCH and MODE==0:
                kb=next_k
                if META==0:next_k=tl.load(COL+begin+pos+1,pos+1<count,0)
                else:next_k=tl.load(IDS+ROW*M+pos+1,pos+1<count,0)
            else:
                if META==0:kb=tl.load(COL+begin+pos)
                else:kb=tl.load(IDS+ROW*M+pos)
            selected=True
        kt=kb*32+kk
        if SPLIT_D:
            if RELOAD_Q:
                q0=tl.load(Q+qt[:,None]*QN+head*QH+d0[None,:],qt[:,None]<N,0)
                q1=tl.load(Q+qt[:,None]*QN+head*QH+d1[None,:],qt[:,None]<N,0)
            k0=tl.load(K+kt[:,None]*KN+head*KH+d0[None,:],kt[:,None]<N,0)
            k1=tl.load(K+kt[:,None]*KN+head*KH+d1[None,:],kt[:,None]<N,0)
            score=tl.dot(q1,tl.trans(k1),tl.dot(q0,tl.trans(k0)))
        else:
            if RELOAD_Q:q=tl.load(Q+qt[:,None]*QN+head*QH+d[None,:],(qt[:,None]<N)&(d[None,:]<96),0)
            k=tl.load(K+kt[:,None]*KN+head*KH+d[None,:],(kt[:,None]<N)&(d[None,:]<96),0)
            score=tl.dot(q,tl.trans(k))
        scale=0.10206207261596575
        if EXP2:scale=0.147244446025901
        score=score*scale
        score=tl.where((qt[:,None]<N)&(kt[None,:]<N)&(qt[:,None]>=kt[None,:])&selected,score,-float('inf'))
        m_new=tl.maximum(m,tl.max(score,1))
        safe=tl.where(m_new==-float('inf'),0.,m_new)
        if EXP2:
            alpha=tl.exp2(m-safe);p=tl.exp2(score-safe[:,None])
        else:
            alpha=tl.exp(m-safe);p=tl.exp(score-safe[:,None])
        l=l*alpha+tl.sum(p,1)
        if SPLIT_D:
            v0=tl.load(V+kt[:,None]*VN+head*VH+d0[None,:],kt[:,None]<N,0)
            v1=tl.load(V+kt[:,None]*VN+head*VH+d1[None,:],kt[:,None]<N,0)
            acc0=acc0*alpha[:,None]+tl.dot(p.to(v0.dtype),v0)
            acc1=acc1*alpha[:,None]+tl.dot(p.to(v1.dtype),v1)
        else:
            v=tl.load(V+kt[:,None]*VN+head*VH+d[None,:],(kt[:,None]<N)&(d[None,:]<96),0)
            acc=acc*alpha[:,None]+tl.dot(p.to(v.dtype),v)
        m=m_new
    if SPLIT_D:
        tl.store(O+qt[:,None]*ON+head*OH+d0[None,:],acc0/l[:,None],qt[:,None]<N)
        tl.store(O+qt[:,None]*ON+head*OH+d1[None,:],acc1/l[:,None],qt[:,None]<N)
    else:tl.store(O+qt[:,None]*ON+head*OH+d[None,:],acc/l[:,None],(qt[:,None]<N)&(d[None,:]<96))


@triton.jit
def _b32_candidate(Q,K,V,O,PTR,COL,IDS,BITS,JOBS,
                   N:tl.constexpr,H:tl.constexpr,M:tl.constexpr,
                   QN:tl.constexpr,QH:tl.constexpr,KN:tl.constexpr,KH:tl.constexpr,VN:tl.constexpr,VH:tl.constexpr,
                   ON:tl.constexpr,OH:tl.constexpr,
                   MODE:tl.constexpr,THRESH:tl.constexpr,STAGES:tl.constexpr,UNROLL:tl.constexpr,
                   META:tl.constexpr,PREFETCH:tl.constexpr,RELOAD_Q:tl.constexpr,EXP2:tl.constexpr,SPLIT_D:tl.constexpr,
                   ORDER:tl.constexpr,PERSIST:tl.constexpr):
    start=tl.program_id(0)
    if PERSIST:
        for job in range(start,H*M,tl.num_programs(0)):
            row=tl.load(JOBS+job) if ORDER else job
            _row(Q,K,V,O,PTR,COL,IDS,BITS,row,N,H,M,QN,QH,KN,KH,VN,VH,ON,OH,
                 MODE,THRESH,STAGES,UNROLL,META,PREFETCH,RELOAD_Q,EXP2,SPLIT_D)
    else:
        row=tl.load(JOBS+start) if ORDER else start
        _row(Q,K,V,O,PTR,COL,IDS,BITS,row,N,H,M,QN,QH,KN,KH,VN,VH,ON,OH,
             MODE,THRESH,STAGES,UNROLL,META,PREFETCH,RELOAD_Q,EXP2,SPLIT_D)


def prepare(counts,indices,order='identity',bits=False):
    l,h,m=counts.shape
    bit_tensor=torch.empty((l,h*m,triton.cdiv(m,32)),device=counts.device,dtype=torch.uint32)
    if bits:_make_bits[(l*h*m,triton.cdiv(m,32))](counts,indices,bit_tensor,m,triton.cdiv(m,32),triton.next_power_of_2(m))
    if order=='descending':jobs=torch.argsort(counts.view(l,-1),dim=-1,descending=True,stable=True).to(torch.int32)
    elif order=='head_descending':
        jobs=(torch.argsort(counts,dim=-1,descending=True,stable=True).to(torch.int32)+torch.arange(h,device=counts.device,dtype=torch.int32)[None,:,None]*m).reshape(l,-1)
    elif order=='ascending':jobs=torch.argsort(counts.view(l,-1),dim=-1,descending=False,stable=True).to(torch.int32)
    elif order=='query_major':jobs=torch.arange(h*m,device=counts.device,dtype=torch.int32).reshape(h,m).T.contiguous().flatten()[None].expand(l,-1)
    else:jobs=torch.empty((l,h*m),device=counts.device,dtype=torch.int32)
    return dict(bits=bit_tensor,jobs=jobs)


def attention(q,k,v,schedule,ids,extra,out,cfg,layer=0):
    n,h,d=q.shape
    assert d==96 and q.stride(-1)==k.stride(-1)==v.stride(-1)==1
    m=triton.cdiv(n,32);persist=cfg.get('persistent',0);order=cfg.get('order','identity')
    grid=h*m if not persist else min(h*m,84*persist)
    return _b32_candidate[(grid,)](q,k,v,out,*schedule,ids,extra['bits'][layer],extra['jobs'][layer],
        n,h,m,*q.stride()[:2],*k.stride()[:2],*v.stride()[:2],*out.stride()[:2],
        cfg.get('mode',0),cfg.get('threshold',101),cfg.get('stages',2),cfg.get('unroll',1),
        cfg.get('metadata',0),cfg.get('prefetch',False),cfg.get('reload_q',False),cfg.get('exp2',False),cfg.get('split_d',False),
        order!='identity',bool(persist),num_warps=cfg.get('warps',4),num_stages=cfg.get('stages',2))
