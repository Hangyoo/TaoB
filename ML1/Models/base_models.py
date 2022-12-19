import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')



class Encoder(nn.Module):
    """
    编码器 Encodes the static & dynamic states using 1d Convolution.
    相当于一个embedding层
    """

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        # input (256,2,11) / (256,2,1)
        output = self.conv(input) # (256,128,11) / (256,128,1)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """
        Calculates attention over the input nodes given the current state.
        将第t步的decoder的输出与encoder中的static_hidden, dynamic_hidden 对比来计算权重,来计算contect vector
        这里涉及到的 v 和 W 分别代表 va 和 Wa
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        # ut = v'tanh(W[xt;ct])

        # torch.nn.Parameter(Tensordata, requires_grad) requires_grad = True 表示可训练
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size), device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        # (256,128,11) batch_size, hidden_size, seq_len
        batch_size, hidden_size, _ = static_hidden.size()

        # Broadcast some dimensions so we can do batch-matrix-multiply
        # 广播一些尺寸，这样我们就可以进行批量矩阵乘法了
        v = self.v.expand(batch_size, 1, hidden_size)  # 扩展为 (256,1,128)
        W = self.W.expand(batch_size, hidden_size, 3 * hidden_size) # 扩展为 (256,128,128*3)

        # decoder_hidden 第t步encoder输出的隐藏层, 与encoder中的static_hidden, dynamic_hidden 对比来计算权重
        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)  # 1--》11 与另两个向量合并 (256,128) --> (256,128，1) --> (256,128,11)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)  # (256,128+128+128,11) == (256,384,11)

        # softmax(attns) 返回的就是权重
        # [0.0908, 0.0909, 0.0911,  ..., 0.0907, 0.0908, 0.0910]
        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden))) # (256,1,128) * (256,128,384) * (256,384,11) --> (256,1,11)
        attns = F.softmax(attns, dim=2)  # (batch, seq_len) 返回的是alignment vector at
        return attns


class Pointer(nn.Module):
    """
        Calculates the next state given the previous state and input embeddings.
        这里涉及到的 v 和 W 分别代表 vc 和 Wc
    """

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        # 计算下一个动作的输出概率
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),  # 2*hidden_size 包含了xt和ct
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        # GRU decoder 网络
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,   # 单层RNN
                          batch_first=True,        # 批量在第一个维度上
                          dropout=dropout if num_layers > 1 else 0)

        self.encoder_attn = Attention(hidden_size) # 返回alignment vector at, 用于计算context vector ct

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        # static_hidden(256,128,11), dynamic_hidden(256,128,11), decoder_hidden(256,128,1), last_hh
        # gru 的输入包含两部分, input and h0; h0不提供时默认为0
        # gru 的返回包含两部分, output and hn
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)  # .transpose 交换2 1 维度
        # rnn_out(256,1,128)  last_hh(1,256,128)
        rnn_out = rnn_out.squeeze(1) # (256,128)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)  # 随机将tensor中的数据编程0，而其他值没有变
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an input context
        # gru 的输出 rnn_out 与encoder 中的static_hidden, dynamic_hidden 对比来计算权重,来计算contect vector
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out) # enc_attn(256,1,11)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats) == (256,1,128)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden) # (256,1,128)-->(256,128,11)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len) == (256,128+128,11)

        v = self.v.expand(static_hidden.size(0), -1, -1) # (1,1,128) --> (256,1,128)
        W = self.W.expand(static_hidden.size(0), -1, -1) # (1,128,256) --> (256,128,256)
        # torch.bmm 矩阵乘法 必须要3维,2维度会报错
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1) # (256,1,11) --> (256,11)  [0.0666, 0.0808, 0.0800,  ..., 0.0910, 0.0915, 0.0836]
        # 在这里没有对probs进行softmax, 后续会结合mark再进行softmax
        return probs, last_hh # (1,256,128)


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
