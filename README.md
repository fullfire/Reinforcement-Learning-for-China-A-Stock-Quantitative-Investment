# Reinforcement-Learning-for-A-Stock-Quant-Investment
  利用中国A股市场的数据训练强化学习Agent。此Agent能够在一个时间窗口内自动择时和控制仓位，依照当前的参数设置，该强化学习Agent的交易行为是中长期交易行为模式。考虑TIME_STEPS个时间窗口的信息，进行投资行为，并在未来的特定时间内进行锁仓操作。本项目使用的强化学习算法为ASAC算法，本文强化学习部分工作主要借鉴[https://github.com/dongminlee94/deep_rl](https://github.com/dongminlee94/deep_rl)。

# Requirements
  1. Pytorch 1.4.0
  2. backtrader
  3. Numpy 1.16.0
  4. Pandas 0.23.4
  
# 强化学习各元素设置（State，Policy，Action，Reward）
  1. 状态： 选取决策步前TIME_STEPS窗口的OHLCV(开盘价，最高价，最低价，收盘价，交易量)作为指标，为了消除个股不同的特性造成的影响，工程中采用的指标都为标准化后的指标（以开盘价为例，开盘价/EMA（开盘价，20））。此外，当前的平均成本和仓位信息也加入了状态中(同样平均成本也需要做标准化处理)，因此在强化学习的训练过程中，其状态是如下的格式 \[ (1, TIME_STEPS, 5),  (1, 2) \]；
  2. 策略：策略网络Trader的输出层激活函数采用tanh（-1，1）；
  3. 行为：假设策略网络输出为p (-1<p<1), 其实际的投资策略为 Min(p,0)* (1 - money_ratio) + Max(0, p) * money_ratio， 其中 money_ratio 为当前状态中的仓位信息，表征当前Agent投资仓位的现金比例。p>0 为继续购买 p * m oney_ratio * all_value 的股票（all_value为当前用户的所有资产总值），p<0 为卖出 -p * (1 - money_ratio) * all_value 等值的股票；
  4. 奖励：目前的奖励包括投资立即收益：锁仓后的总值/锁仓前的总值 - 1， 此为锁仓前投资行为的收益。机会成本，即，如果当前Agent不做任何投资行为(p=0)，其收益为机会成本；Agent投资行为的即时奖励为 w_1 * 立即收益 - w_2 * 机会成本；

# 训练数据
  本策略只采用OHLCV信息，训练数据可以从tushare，baostock等数据源获取；data文件夹存放了样例数据供参考；

# 运行
  1. 准备Requirements的环境；
  2. 运行 asac_main.py
  本项目提供的一个样例模型的结果表现如下（在Nvidia Geforce RTX 2080Ti训练了6小时）：
  ![image](https://github.com/SchindlerLiang/Reinforcement-Learning-for-A-Stock-Quant-Investiment/blob/master/reward.png)
  对于大部分股票，其在最近300个交易日的收益大于0，14只股票的收益平均值为4.8%。
  
  如下是一个近300天呈下跌趋势的股票的交易结果（开局80万，装备全靠割韭菜）：
  ![image](https://github.com/SchindlerLiang/Reinforcement-Learning-for-A-Stock-Quant-Investiment/blob/master/transaction.png)
  可以看出即使在全年下跌趋势中，该智能体Agent依然能够抓住几次局部反弹的机会赚取投资利润。从上图也明显看出，该策略的股票购买时机把握较为精准，但是卖出时机还有较大的改进空间。
  
  训练：
  1. 从tushare、baostock获取股票OHLCV等信息；
  2. 修改强化学习的模型参数，TIME_STEPS, COLD_DAY等参数；
  3. 修改asac_main.py中trader的 eval_mode=False；
  4. 运行asac_main.py；
  
  Note：由于当前ASAC算法的网络参数选取较小，当前的架构对于GPU没有强制需求。当前主要的问题受限于backtrader的回测速度。从强化学习的训练过程来看，当前的状态还远没有收敛，后续会陆续发布新训练的weights文件。
  当前强化学习的训练过程包含无数backtrader的回测过程，在训练过程中可以追踪各指标的变化，从而设计不同的参数组合进行优化。




  
