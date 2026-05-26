已检查 `src/data/imagenet_dataset.py` 中的 train_loader
当前 DataLoader 使用：
shuffle=True
未发现sampler=ClassBalancedSampler(...)
因此：
ClassBalancedSampler 已实现
当前未接入训练 dataloader，训练仍采用标准随机采样
关于暂不启用 ClassBalancedSampler 的技术说明：
现状排查：经排查 src/data/imagenet_dataset.py，当前 train_loader 采用了标准的随机采样策略（shuffle=True）。尽管 ClassBalancedSampler 已在代码中实现，但当前阶段并未接入训练数据流。
数据集不平衡度评估：当前 ImageNet-1K 子集的最大/最小类别样本比例仅约为 1.78x，属于极轻微的类别不平衡。
技术抉择论证：
过拟合风险：在极低的不平衡比（IR < 2）下，强行应用类别平衡采样（如过采样）会导致尾部样本被频繁重复学习，这极易引发模型对少数类样本的像素级过拟合，反而降低在测试集上的泛化能力。
训练稳定性：非均匀采样会扰乱自然的数据分布，对于轻度不平衡的数据集，这可能会引发梯度波动的副作用，降低训练收敛的稳定性。
结论：当前阶段保留 shuffle=True 的标准经验风险最小化（ERM）采样策略是最稳妥且高效的选择，暂不启用 ClassBalancedSampler。