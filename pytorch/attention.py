import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q,K,V,mask=None):
    # Q：[B, H, L_q, d_k]
    # K: [B, H, L_k, d_k]
    # V: [B, H, L_k, d_v]

    # 1. 打分
    scores=torch.matmul(Q,K.transpose(-2,-1))

    # 2. 压缩
    d_k=Q.size(-1)
    scores=scores/torch.sqrt(torch.tensor(d_k,dtype=torch.float32))

    # 3. mask(padding或者causal)
    if mask is not None:
        scores=scores.masked_fill(mask==0,float('-inf'))
    
    # 4. softmax 归一化
    attn=F.softmax(scores,dim=-1)

    # 5. 加权求和
    output=torch.matmul(attn,V)
    return output,attn

if __name__=='__main__' :
    # =============================
    # 测试部分
    # =============================
    B, H, L_q, L_k, d_k, d_v = 2, 2, 3, 4, 8, 8  # batch=2, head=2, q_len=3, k_len=4, dim=8

    Q = torch.randn(B, H, L_q, d_k)
    K = torch.randn(B, H, L_k, d_k)
    V = torch.randn(B, H, L_k, d_v)
    # 例子：构造一个 padding mask (假设最后一个 token 是 padding)
    mask = torch.ones(B, 1, 1, L_k)   # [B,1,1,L_k]
    mask[:, :, :, -1] = 0             # 最后一列屏蔽掉

    output, attn = scaled_dot_product_attention(Q, K, V, mask)

    print("Q:", Q.shape)
    print("K:", K.shape)
    print("V:", V.shape)
    print("attn:", attn.shape)
    print("output:", output.shape)

    # 看一下 softmax 后的权重是否屏蔽成功（最后一列应该接近 0）
    print("attention weights for head 0, batch 0:\n", attn[0,0])