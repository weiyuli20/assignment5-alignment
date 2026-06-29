1. 累计期望奖励和优化目标

累计期望奖励：

是每条样本内所有 token 的 logprob 先求和，再乘以这条轨迹的总回报 R，最后把全部样本加起来取平均，才是梯度估计。

1. 先分清两个基础概念
2. （1）一条轨迹所有动作联合概率一条轨迹 \(\tau=(s_0,a_0,s_1,a_1,...,s_T,a_T)\) 的完整概率：
\(P(\tau) = \prod_{t=0}^T \pi_\theta(a_t|s_t)\)
概率是相乘；
取对数后，乘积变加法：
\(\log P(\tau) = \sum_{t=0}^T \log\pi_\theta(a_t|s_t)\)
👉 一条轨迹整体对数概率 = 每一步 token 的 logprob 相加。

（2）策略梯度的核心项（不是单纯求和 logprob）原始 REINFORCE 梯度单条轨迹贡献：
\(\left(\sum_{t=0}^T \log\pi_\theta(a_t|s_t)\right) \cdot R(\tau)\)

流程：

单条生成里，每个 token 算 \(\log\pi\)，全部加总；

乘上这条完整句子的总分回报 \(R(\tau)\)；

一批 N 条样本全部累加，除以 N 做平均，得到梯度估计 \(\hat g\)。


优化目标：

因为是梯度上升算法，最大化累计期望奖励

因此在梯度上升优化的过程中，一条轨迹的奖励值大时，会增加生成这条轨迹的概率，奖励值小的轨迹会被抑制生成


优势函数

因为奖励的方差比较大，所以采用一些方式减小方差


on-policy

训练数据由当前正在优化的同一个策略收集得到，REINFORCE就是一种on_policy 算法

1. 从当前策略 πθ 中采样一批轨迹（rollouts）{ τ ( i ) } i = 1 N \{\tau^{(i)}\}_{i=1}^{N}{τ (i) } i=1N

2. 用这些轨迹来近似策略梯度 

3. 使用该梯度更新策略参数 θ ← θ + α g ^ \theta \leftarrow \theta + \alpha \hat{g}θ←θ+α 

** 为了获得一小步梯度更新，我们需要进行大量推理来采样新的轨迹批次，然而，语言模型的行为通常不会在一次更新中发生显著变化，因此这种 on-policy 方法效率非常低。**



off-policy

使用的轨迹（rollouts）并不是来自当前正在优化的策略，而是来自另一个策略。像 PPO 和 GRPO 这样的主流策略梯度算法的离策略变体，会使用旧策略 πold 生成的轨迹来优化当前策略 πθ

grpo

组相对策略优化  无需训练价值网络

一个问题采样多条输出，计算每条rollout的奖励  ，然后在组内计算 组归一化奖励
​
1. 实现组归一化奖励
   ```
   uv run pytest -k test_compute_group_normalized_rewards
```

2. 计算逐token策略梯度，即每个预测token的log prob * (奖励或优势，是句子级别的）
```
uv run pytest -k test_compute_naive_policy_gradient_loss
```

3. GRPO clip :逐token的策略梯度裁剪  策略模型和old模型计算一个比值ratio  **old是什么呢？**
```
uv run pytest -k test_compute_grpo_clip_loss

```

4. 3种策略梯度损失对比

1). 无baseline和组归一化, 使用原始奖励 *logprob

2).无baseline,计算组归一化奖励

3).有baseline ,grpo_clip裁剪方式，计算一个ratio * logprob


