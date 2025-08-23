import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x: [B, L, d_model]
        mean = x.mean(dim=-1, keepdim=True)                # [B, L, 1]
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # [B, L, 1]

        x_norm = (x - mean) / torch.sqrt(var + self.eps)   # 归一化
        output = self.gamma * x_norm + self.beta           # 缩放 + 平移

        return output

if __name__ == "__main__":
    B, L, d_model = 2, 4, 8
    x = torch.randn(B, L, d_model)

    ln = LayerNorm(d_model)
    output = ln(x)

    print("input:", x.shape)
    print("output:", output.shape)

    # 检查均值和方差（应该接近 0 和 1）
    mean = output.mean(dim=-1)
    var = output.var(dim=-1, unbiased=False)
    print("mean per token:", mean)
    print("var per token:", var)
