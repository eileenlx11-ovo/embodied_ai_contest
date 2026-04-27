# TODO: Label Smoothing Loss (手动实现版)
# 公式: loss = (1-ε) * CE(one_hot, p) + ε * CE(uniform, p)
# 注: PyTorch 内置 CrossEntropyLoss(label_smoothing=0.1) 已在 base_loss.py 中使用
