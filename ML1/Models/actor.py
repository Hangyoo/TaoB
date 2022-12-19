import torch
import torch.nn as nn
import torch.nn.functional as F
from ML1.Models.base_models import Encoder, Pointer, Attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DRL4TSP(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    实际上这个就是Actor网络(指针网络)
    Encoder: 1D-conv (input_size, Output_size=128, kernel size=1, stride=1)
    Decoder: Gru(hiddenz-size=128, num_layer=1) + Attention

    Parameters
    ----------
    static_size: int
    定义模型的静态元素中有多少个特征 (例如：(x, y)坐标为2)
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        有动态特征即输入动态特征,对于TSP等无动态特征问题,输0向量即可
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    定义模型的动态元素中有多少个特征
        (例如，对于具有（负荷，需求）属性的VRP来说，是2。TSP没有
        有动态元素，但为了确保与其他优化问题的兼容性
        问题的兼容性，假设我们只是传入一个零的向量。
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
        定义了RNN隐藏层维度
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
        如果提供，该方法用于计算输入的动态元素如何更新,并在输入元素的每个 "点 "之后被调用。
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
        允许我们指定输入序列的哪些元素被允许被选中。这对于加快网络的训练非常有用。
        通过为算法提供一种 "规则 "的指导。如果没有提供掩码 的情况下，
        我们会在一个固定的迭代次数后终止搜索 以此来避免永远延长搜索的时间
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
        指定在解码器RNN中使用的隐藏层的数量
    dropout: float
        Defines the dropout rate for the decoder
        解码器的dropout率
    """
    def __init__(self, static_size, dynamic_size, hidden_size,update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        # static_size=2, dynamic_size=2, hidden_size=128
        super(DRL4TSP, self).__init__()

        # dynamic_size 的维度至少要大于1
        if dynamic_size < 1:
            # 即使没有动态特征, dynamic_size长度也要大于0
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn  # update_dynamic
        self.mask_fn = mask_fn      # update_mask

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)   # static_size=2 转换为 hidden_size=128
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size) # static_size=2 转换为 hidden_size=128
        self.decoder = Encoder(static_size, hidden_size)          # static_size=2 转换为 hidden_size=128
        self.pointer = Pointer(hidden_size, num_layers, dropout)  # 返回 probs, last_hh

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)  # 基本思想是通过网络层时，输入和输出的方差相同，包括前向传播和后向传播

        # Used as a proxy initial state in the decoder when not specified
        # 当没有指定时，在解码器中作为代理初始状态使用
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device) # size(1,2,1)

    def forward(self, static, dynamic, decoder_input=None, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats:特征数=2, num_cities:城市数)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
            定义了要考虑的静态元素。对于TSP来说，这可能是像(x, y)坐标这样的东西，它们一直不会改变。
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
            定义了要考虑的静态元素。对于VRP来说，这可以是像每个城市的（负荷，需求）。如果没有动态元素，可以将其设置为None。
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
            定义了解码器的输出。目前，我们只使用静态元素（例如（x，y）坐标），但技术上也可以是其他东西
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
            定义了RNN的最后一个隐藏状态
        """

        batch_size, input_size, sequence_size = static.size()  # (256,2,11) (batch_size, feats:特征数=2, num_cities:城市数)
        if decoder_input is None:
            # 函数返回张量在某一个维度扩展之后的张量，就是将张量广播到新形状
            decoder_input = self.x0.expand(batch_size, -1, -1)  # 增加一个维度 扩展到 batch_size

        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device=device) #(256,11) mask向量: batch_size行 sequence_size列 的全1向量

        # 预测开始
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is not None else 1000  # 没有设置时，在1000步停止
        # max_steps = sequence_size if self.mask_fn is None else 1000  # 没有设置时，在1000步停止

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        # 静态元素只需要处理一次，并且可以在所有的 'pointing' 迭代中使用。
        # 当/如果动态元素发生变化时，它们的表现形式将需要再次得到计算。
        static_hidden = self.static_encoder(static)     # (256,128,11) 嵌入层
        dynamic_hidden = self.dynamic_encoder(dynamic)  # (256,128,11) 嵌入层

        for _ in range(max_steps): # max_steps = 11

            if not mask.byte().any():
                break

            # ... but compute a hidden rep for each element added to sequence
            decoder_hidden = self.decoder(decoder_input)

            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden,
                                          last_hh)

            # mask 为全1向量,那么log10(1)=0,不影响probs
            probs = F.softmax(probs + mask.log(), dim=1) # 加mask后进行softmax操作 (256,11)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            # 在训练时,下一步根据概率随机采样
            # 在测试时,下一步根据贪婪方法选择最大值
            if self.training:
                # 其作用是创建以参数probs为标准的类别分布，样本是来自“0，...，K-1”的整数，K是probs参数的长度。
                # 也就是说，按照probs的概率，在相应的位置进行采样，采样返回的是该位置的整数索引。
                m = torch.distributions.Categorical(probs)
                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                ptr = m.sample()  # (256) ptr是位置的整数索引
                while not torch.gather(input=mask, dim=1, index=ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                # 可以根据当前的policy π 的概率密度函数采样得到at, 然后计算采样后得到的-logπ(at|st)
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy 贪婪方法
                logp = prob.log()

            # After visiting a node update the dynamic representation
            # 到访一个点后需要对dynamic representation进行更新
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)  # 对新状态进行embedding

                # Since we compute the VRP in minibatches, some tours may have
                # number of stops. We force the vehicles to remain at the depot 
                # in these cases, and logp := 0
                # 由于我们是以小批量的方式计算VRP，所以有些线路可能会有一些停靠点。在这些情况下，我们强制车辆留在仓库，logp=0
                is_done = dynamic[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)

            # And update the mask so we don't re-visit if we don't need to
            # 更新mask, 因为不想重复访问一个点
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data).detach() # 返回一个新Tensor 从计算图中脱离出来，无需计算梯度

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

            decoder_input = torch.gather(static, 2,
                                         ptr.view(-1, 1, 1)
                                         .expand(-1, input_size, 1)).detach()
        # 将tour_idx,tour_logp列表转换为torch
        tour_idx = torch.cat(tour_idx, dim=1)  # (256,11) (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)
        return tour_idx, tour_logp