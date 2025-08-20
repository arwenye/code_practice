import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model%num_heads==0,"d_model 必须被 num_heads 整除"  #断言语句，不满足条件时会抛出AssertionError
        self.num_heads=num_heads

        # 计算各头的维度
        self.d_head=d_model//num_heads

        # Q，K，V的线性变换层
        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)

        # 输出投影层
        self.W_o=nn.Linear(d_model,d_model)

    def forward(self, x, mask=None):
        batch_size,seq_len,d_model=x.shape
        print(f"输入: {x.shape}")  # [B, L, D]

        # 1. 计算 Q, K, V
        Q=self.W_q(x) # [B, L, D]
        K=self.W_k(x)
        V=self.W_v(x)
        print(f"Q/K/V: {Q.shape}")

        # 2. 拆分成多头，并转置方便矩阵乘法
        # [B, L, D] -> [B, num_heads, L, d_head]
        def reshape_heads(tensor):
            return tensor.view(batch_size,seq_len,self.num_heads,self.d_head).transpose(1,2)
        Q=reshape_heads(Q) # [B, h, L, d_head]
        K=reshape_heads(K)
        V=reshape_heads(V)
        print(f"Q 拆头后: {Q.shape}")

        # 3. 计算 Attention score: QK^T / sqrt(d_head)
        # Q: [B, h, L, d_head]
        # K.transpose(-2,-1): [B, h, d_head, L]
        scores = torch.matmul(Q,K.transpose(-2,-1))/(self.d_head**0.5)
        print(f"Attention scores: {scores.shape}")  # [B, h, L, L]
        
        # ======== 加 Mask ========
        if mask is not None:
            # mask: [B, 1, 1, L] 或 [B, 1, L, L]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores,dim=-1) # [B, h, L, L ]
        print(f"Attention 权重: {attn.shape}")
        

        # 4. 加权求和
        context = torch.matmul(attn,V) #[B, h, L, d_head]
        print(f"context: {context.shape}")


        # 5. 拼接所有头
        context = context.transpose(1,2).contiguous().view(batch_size,seq_len,d_model)
        print(f"拼接后: {context.shape}")


        # 6. 输出投影
        out = self.W_o(context)
        print(f"最终输出: {out.shape}")

        return out




if __name__=='__main__':
    batch_size, seq_len, d_model, num_heads = 2, 5, 16, 4
    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadSelfAttention(d_model, num_heads)
    out = mha(x)
    print(out.shape)  # 期望 [2, 5, 16]