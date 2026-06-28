1. 实现 adapter里的 tokenize_prompt_and_output 方法，输入prompt和answer 能够输出input_ids,labels, response_mask

```
uv run pytest -k test_tokenize_prompt_and_output
```
通过测试

2. 计算 per_token 的entropy  ,表示模型对每个输出token的自信程度
交叉熵损失计算  -sum(p*log_p)



3. 计算模型预测出的每个答案标签的概率

从模型输出的logits->计算log_softmax, 选出仅答案id位置上的概率


per_token_entropy是计算自信程度的，不能混淆


4. sft loss 
负对数似然

上一步已经计算出了模型分配给真实目标token的 log_prob(log_softmax)  形状为（B,seq_len）

现在需要累加且仅考虑mask=1的位置

然后取负号，得到loss, 最小化损失，等于最大化目标token的log_prob


5.专家迭代
专家迭代：根据训练数据集 （利用当前的策略模型每条样本推理出多个轨迹），根据一定规则，筛选过滤样本，保留下的数据加入训练集，利用这个新的训练集训练模型得到新的策略  直至迭代这个过程到指定step
