import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # [B, L, d_model]
        print("input:", x.shape)
        x = self.W1(x)           # [B, L, d_ff]
        print("after W1:", x.shape)
        x = self.relu(x)         # [B, L, d_ff]
        x = self.dropout(x)      # [B, L, d_ff]
        x = self.W2(x)           # [B, L, d_model]
        print("after W2:", x.shape)
        return x


if __name__ == "__main__":
    B, L, d_model, d_ff = 2, 5, 16, 64   # batch=2, 序列长=5, 模型维度=16, FFN维度=64
    x = torch.randn(B, L, d_model)       # [2, 5, 16]

    ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.1)
    output = ffn(x)

    print("output:", output.shape)       # 应该是 [2, 5, 16]