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
