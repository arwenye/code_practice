import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super(SimpleMLP, self).__init__()
        # TODO: 定义两层 Linear，添加 Dropout 和激活函数
        self.l1=nn.Linear(input_dim,hidden_dim)
        self.l2=nn.Linear(hidden_dim,output_dim)
        self.relu=nn.ReLU()
        if dropout>0: #方便eval时不调用dropout
            self.drop=nn.Dropout(dropout)
        else:
            self.drop=nn.Identity() # 等价于啥都不做 ✅直接返回

    def forward(self, x):
        # TODO: 完成前向传播，顺序为：Linear -> ReLU -> Dropout -> Linear
        print(f'[Input] x.shape={x.shape}')#做注释方便后续调试
        x=self.l1(x)
        print(f'[After Linear1] x.shape={x.shape}')
        x=self.relu(x)
        print(f'[After Relu] x.shape={x.shape}')
        x=self.drop(x)
        print(f'[After Drop] x.shape={x.shape}')
        x=self.l2(x)
        print(f'[After Linear2] x.shape={x.shape}')
        return x


if __name__=='__main__':
    model = SimpleMLP(input_dim=128, hidden_dim=64, output_dim=10)
    x = torch.randn(32, 128)  # batch_size=32
    out = model(x)
    print(out.shape)  # 应该是 [32, 10]
