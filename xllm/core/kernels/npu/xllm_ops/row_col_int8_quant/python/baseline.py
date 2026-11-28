"""Same-semantics PyTorch NPU compositions. No custom kernel calls."""
import torch
import torch_npu
import torch.nn.functional as F
OP_INDEX=2

def candidates(inputs,attrs):
    if OP_INDEX==1:
        x,g1,b1,g2,b2=inputs
        eps=attrs.get("epsilon",1e-5)
        def two_layer_norms():
            return F.layer_norm(x,(x.shape[1],),g1,b1,eps),F.layer_norm(x,(x.shape[1],),g2,b2,eps)
        def shared_statistics():
            a=x.float(); centered=a-a.mean(dim=1,keepdim=True)
            z=centered*torch.rsqrt((centered*centered).mean(dim=1,keepdim=True)+eps)
            return (z*g1.float()+b1.float()).to(x.dtype),(z*g2.float()+b2.float()).to(x.dtype)
        return {"two_layer_norms":two_layer_norms,"shared_fp32_statistics":shared_statistics}
    if OP_INDEX==2:
        x=inputs[0]
        def row_col_composition():
            a=x.float(); absolute=a.abs()
            r=absolute.amax(dim=1); c=absolute.amax(dim=0)
            def quant(scale):
                safe=torch.where(scale==0,torch.ones_like(scale),scale)
                return (a*(127.0/safe)).round().clamp(-127,127).to(torch.int8)
            return quant(r[:,None]),quant(c[None,:]),r,c
        def two_dynamic_quant():
            qr,_=torch_npu.npu_dynamic_quant(x)
            qc,_=torch_npu.npu_dynamic_quant(x.t().contiguous())
            absolute=x.float().abs()
            return qr,qc.t().contiguous(),absolute.amax(dim=1),absolute.amax(dim=0)
        return {"fp32_absmax_round_cast":row_col_composition,"two_dynamic_quant_transpose":two_dynamic_quant}
    qkv,pk,pv=inputs
    b,h,l,d=pk.shape
    def views_and_concatenation():
        parts=qkv.reshape(b,3,h,d)
        q=parts[:,0].reshape(b,h,1,d).clone(memory_format=torch.contiguous_format)
        k=parts[:,1].reshape(b,h,1,d)
        v=parts[:,2].reshape(b,h,1,d)
        return q,torch.cat((pk,k),dim=2),torch.cat((pv,v),dim=2)
    return {"views_concat_q_copy":views_and_concatenation}
