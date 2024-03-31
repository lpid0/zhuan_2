from models.siamese import Siamese
# 创建一个 Siamese 模型实例
import torch

# 实例化 Siamese 模型
model = Siamese()

# 加载预训练模型的权重（如果有的话）
model.load_state_dict(torch.load("checkpoints/best.pt"))

# 设置模型为评估模式
model.eval()

# 创建虚拟的输入数据
# 假设 left 和 right 是两张输入图片的张量，大小为 (batch_size, channels, height, width)
left = torch.randn(1, 3, 32, 32)  # 示例输入数据，大小为 32x32，3 个通道
right = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, (left, right), "siamese.onnx", verbose=True)

