import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Models.actor import DRL4TSP
from Tasks import vrp
from Tasks.vrp import VehicleRoutingDataset
from Models.critc import StateCritic

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled=False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Detected device {}'.format(device))


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=5):
    """
    Used to monitor progress on a validation set & optionally plot solution.
    用于监测验证集的进展
    """

    #eval()的作用是不启用BatchNormalization和Dropout，并且不会保存中间变量、计算图
    #训练完train_datasets之后，model要来测试样本了。在model(test_datasets)之前，需要加上model.eval().
    # 否则的话，有输入数据，即使不训练，它也会改变权值。
    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        # 则主要是用于停止autograd模块的工作，以起到加速和节省显存的作用。
        # 它的作用是将该with语句包裹起来的部分停止梯度的更新，从而节省了GPU算力和显存，但是并不会影响dropout和BN层的行为
        # 计算验证集上的路径
        with torch.no_grad():
            tour_indices, tour_logp = actor.forward(static, dynamic, x0)

        # 计算奖励
        # item方法是得到只有一个元素张量里面的元素值(无论元素维度如何)
        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards) # 返回在整个验证集上的平均奖励(真实)值


def train(actor, critic, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()  # datetime.time(10, 44, 7, 518779)
    now = now.replace(':', '_') # '10:44:30.393527' --> '10_44_30.393527'
    save_dir = os.path.join(task, '%d' % num_nodes, now) # 保存文件的位置

    print('Starting training')

    # 没有文件创建文件,后续保存参数值
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)    # actor  网络的优化器
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr) # critic 网络的优化器

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, shuffle=False, num_workers=0)

    best_params = None    # 最优参数
    best_reward = np.inf  # 最优回报

    for epoch in range(20):

        actor.train()
        critic.train()

        # 运行时间, 损失, 真实奖励(旅行的路径距离), critic网络的估计奖励
        times, losses, rewards, critic_rewards = [], [], [], []

        # 开始计时
        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch  # [(256,2,11),(256,2,11),(256,2,1)]

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            # actor:就是指针网络, 计算预测路径并返回logP值用于计算loss
            tour_indices, tour_logp = actor(static, dynamic, x0) # tour_indices(256,11) 路径

            # Sum the log probabilities for each city in the tour
            # 计算路径真实值
            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            # critic网络估计reward
            critic_est = critic(static, dynamic).view(-1)  # critic估计的奖励

            # 前馈 - 反馈 - 更新
            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))  # actor_loss
            actor_optim.zero_grad()
            actor_loss.backward()
            # 这个函数的主要目的是对parameters里的所有参数的梯度进行规范化
            # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值，如果梯度超过阈值，那么就截断，将梯度变为阈值
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_loss = torch.mean(advantage ** 2)  # critic_loss
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end  # 每隔100个batch返回一个时间

                mean_loss = np.mean(losses[-100:])     # 当前100个批次的actor平均损失
                mean_reward = np.mean(rewards[-100:])  # 平均真实奖励

                print(' Batch %d/%d, reward: %2.3f, loss: %2.4f, time: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss, times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # 保存Actor和Critic的权重
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set tours
        # 在验证集上训练
        valid_dir = os.path.join(save_dir, '%s' % epoch)
        mean_valid = validate(valid_loader, actor, reward_fn, render_fn,valid_dir, num_plot=5)

        # 保存最好的模型参数
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        # 输出在一个epoch下: loss 和 reward(真实) reward(valid); 本epoch的执行时间; 平均100个batch的运行时间
        print('Mean epoch loss: %2.4f, reward(true):%2.4f, reward(valid):%2.4f, took: %2.4fs (%2.4fs / 100 batches)\n'\
              %(mean_loss, mean_reward, mean_valid, time.time() - epoch_start,np.mean(times)))


def train_vrp(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23 (Greedy)

    print('Starting VRP training')

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}  # 不同城市对应的装载量
    MAX_DEMAND = 9  # 最大需求
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 2 # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    # 训练集
    train_data = VehicleRoutingDataset(args.train_size,  # 1e6
                                       args.num_nodes,   # 默认城市数量 10
                                       max_load,         # 城市最大需求
                                       MAX_DEMAND,       # 车辆最大负载
                                       args.seed)        # 随机种子

    # 验证集
    valid_data = VehicleRoutingDataset(args.valid_size,  # 1e3
                                       args.num_nodes,   # 默认城市数量 10
                                       max_load,         # 城市最大需求
                                       MAX_DEMAND,       # 车辆最大负载
                                       args.seed + 1)    # 随机种子


    # actor网络
    actor = DRL4TSP(STATIC_SIZE,       # 2
                    DYNAMIC_SIZE,      # 2
                    args.hidden_size,  # 128
                    train_data.update_dynamic,
                    train_data.update_mask,
                    args.num_layers,   # 1
                    args.dropout).to(device) # 0.1

    print('Actor: {} '.format(actor))

    # criti网络
    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    print('Critic: {}'.format(critic))


    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vrp.reward
    kwargs['render_fn'] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt') # .pt 保存张量
        actor.load_state_dict(torch.load(path, device))  # 用于将预训练的参数权重加载到新的模型之中

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device)) # 用于将预训练的参数权重加载到新的模型之中

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

    print('Average tour length: ', out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_nodes', default=10, type=int)  # 城市数目
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=1000000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)

    args = parser.parse_args()
    print(args)

    #print('NOTE: SETTTING CHECKPOINT: ')
    #args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    #print(args.checkpoint)

    train_vrp(args)
