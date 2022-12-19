"""Defines the main task for the VRP.

The VRP is defined by the following traits:
    1. Each city has a demand in [1, 9], which must be serviced by the vehicle
    2. Each vehicle has a capacity (depends on problem), the must visit all cities
    3. When the vehicle load is 0, it __must__ return to the depot to refill
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class VehicleRoutingDataset(Dataset):
    def __init__(self, num_samples, input_size, max_load=20, max_demand=9, seed=None):
        # num_samples 1e3 - 1e6
        # input_size 默认城市数量 10
        # max_load 城市最大需求
        # max_demand 车辆最大负载
        # seed 随机种子
        super(VehicleRoutingDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples  # 生成的样本个数
        self.max_load = max_load # 汽车最大运输能力
        self.max_demand = max_demand  # 城市最大需求

        # Depot location will be the first node in each
        # 10个城市并不包括depot,所以长度是城市数目+1
        locations = torch.rand((num_samples, 2, input_size + 1))  # (1000000,2,11) num_samples个 2行  城市数+1列  介于(0,1)之间的坐标
        self.static = locations

        # All states will broadcast the drivers current load
        # Note that we only use a load between [0, 1] to prevent large
        # numbers entering the neural network 只使用[0，1]之间的负载，以防止大数字进入神经网络。
        dynamic_shape = (num_samples, 1, input_size + 1)
        loads = torch.full(dynamic_shape, 1.) # num_samples个 1行  城市数+1列  值为1的数

        # All states will have their own intrinsic demand in [1, max_demand), 
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30, 
        # demands will be scaled to the range (0, 3)
        demands = torch.randint(1, max_demand + 1, dynamic_shape)
        demands = demands / float(max_load) # 需求介于 0.05 - 0.5

        demands[:, 0, 0] = 0  # 所有始发点(depot)需求设置为0
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))  # torch.Size([1000000, 2, 11])
        # print(self.dynamic.shape)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static:访问城市的坐标, dynamic:当前[loads, demand], start_loc:起始点坐标)
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1])

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len) (1,2,11)
        """

        # Convert floating point to integers for calculations
        loads = dynamic.data[:,0,:]  # (batch_size, seq_len) (256,11)
        demands = dynamic.data[:,1,:]  # (batch_size, seq_len) (256,11)

        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        # 若没有正数需求，则结束旅行
        # 注意: 第一个点是depot, 它总是负需求

        # 当所有城市的需求都是0时, 结束程序
        if demands.eq(0).all():
            # 将demand中的元素逐个与0比较,相等返回True 否则返回False
            # .all() 所有元素都是True时候返回True

            # 结束
            return demands * 0.

        # Otherwise, we can choose to go anywhere where demand is > 0
        # 否则，可以到任意一个 demand > 0 的城市
        # demands 元素与0不等, 相应的位置返回1  demands.lt(loads) 比较 demands < loads, 成立相应位置返回1
        # new_mask 表示了可以访问的点 (既有需求，且车辆剩余能覆盖该城市需求)
        new_mask = demands.ne(0) * demands.lt(loads)  # (256,11)

        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0)  # 等于0返回False  否则返回True

        if repeat_home.any():
            idx = repeat_home.nonzero()  # repeat_home.nonzero() 输出非零元素坐标
            new_mask[idx, 0] = 1.
        if (~repeat_home).any():   # (1 - repeat_home) 对布尔变量进行调整
            new_mask[(~repeat_home).nonzero(), 0] = 0.

        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float()     # (256,11)
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()  #(256,11)

        combined = (has_no_load + has_no_demand).gt(0)  # 比较 (has_no_load + has_no_demand)>0, 成立返回1, 否则返回0
        if combined.any():
            # print(new_mask)
            new_mask[combined.nonzero(), 0] = 1.  # 不允许回起点
            new_mask[combined.nonzero(), 1:] = 0. # 有需求的点不mask

        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx):
        """
            Updates the (load, demand) dataset values.
            chosen_idx: 256批量, 经过 pointer_net 输出的要访问的下一个点
            dynamic: (256,2,11) 第一个为load，第二个为demand
        """

        # Update the dynamic elements differently for if we visit depot vs. a city
        # .ne(0): !=0 返回True; ==0 返回True  0点代表着depot点
        visit = chosen_idx.ne(0) # 需要访问的点
        depot = chosen_idx.eq(0) # depot点

        # Clone the dynamic variable so we don't mess up graph
        # 复制动态变量，这样我们就不会弄乱图形了
        all_loads = dynamic[:, 0].clone()   # (256,11)
        all_demands = dynamic[:, 1].clone() # (256,11)

        # torch.gather(input, dim, index)  dim索引的轴, index索引
        # dim = 1 表示在横向，所以索引就是列号
        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1)) # 对数据维度进行扩充 chosen_idx (256) --> (256,1); load (256,1)
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        # 如果我们选择访问一个城市，尽量满足 尽可能多的需求
        if visit.any():
            # torch.clamp(input, min, max) input中的值不能超过上届，也不能超过下界
            new_load = torch.clamp(load - demand, min=0)    # 更新小车的load
            new_demand = torch.clamp(demand - load, min=0)  # 更新城市的demand

            # Broadcast the load to all nodes, but update demand seperately
            # 将小车的load广播到每个点，但对于每个城市的demand要分别更新
            visit_idx = visit.nonzero().squeeze()  # visit.nonzero() 如 torch.Size([228, 1]) 压缩维度, 去掉1的维度
            all_loads[visit_idx] = new_load[visit_idx]

            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = new_load[visit_idx].view(-1) - 1.

        # Return to depot to fill vehicle load
        # 返回起点重新装载
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.

        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        return torch.as_tensor(tensor.data, device=dynamic.device)


def reward(static, tour_indices):
    """
    用路径的距离作为奖励函数
    Euclidean distance between all cities / nodes given by tour_indices
    static: (256,2,11)      每个点的坐标位置
    tour_indices: (256,11)  每个tour轨迹
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1) # (256,11) --> (256,1,11) --> (256,2,11)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1) # (256,11,2) 对应具体路径的坐标

    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0
    # static.data[:, :, 0].size() ——> (256,2)  获取depot的位置
    start = static.data[:, :, 0].unsqueeze(1) # (256,1,2)
    y = torch.cat((start, tour, start), dim=1) # (256,13,2)

    # Euclidean distance between each consecutive point
    # 计算出每两个点之间的欧式距离
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1,:] - y[:, 1:,:], 2), dim=2))  # (256,12); dim=2按列求和,把x,y都加起来

    # 对所有点之间的距离相加
    return tour_len.sum(dim=1)


def render(static, tour_indices, save_path):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)


'''
def render(static, tour_indices, save_path):
    """Plots the found solution."""

    path = 'C:/Users/Matt/Documents/ffmpeg-3.4.2-win64-static/bin/ffmpeg.exe'
    plt.rcParams['animation.ffmpeg_path'] = path

    plt.close('all')

    num_plots = min(int(np.sqrt(len(tour_indices))), 3)
    fig, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                             sharex='col', sharey='row')
    axes = [a for ax in axes for a in ax]

    all_lines = []
    all_tours = []
    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        cur_tour = np.vstack((x, y))

        all_tours.append(cur_tour)
        all_lines.append(ax.plot([], [])[0])

        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

    from matplotlib.animation import FuncAnimation

    tours = all_tours

    def update(idx):

        for i, line in enumerate(all_lines):

            if idx >= tours[i].shape[1]:
                continue

            data = tours[i][:, idx]

            xy_data = line.get_xydata()
            xy_data = np.vstack((xy_data, np.atleast_2d(data)))

            line.set_data(xy_data[:, 0], xy_data[:, 1])
            line.set_linewidth(0.75)

        return all_lines

    anim = FuncAnimation(fig, update, init_func=None,
                         frames=100, interval=200, blit=False,
                         repeat=False)

    anim.save('line.mp4', dpi=160)
    plt.show()

    import sys
    sys.exit(1)
'''
