# TODO: 样本选择机制
# 策略一: LossThresholdSelector — 基于 loss 阈值筛选干净样本
# 策略二: ConfidenceSelector — 基于 softmax 置信度筛选
# 接口: def select(self, losses, epoch) -> List[int]
