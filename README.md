# LetNet-5
利用letnet5框架深度学习手写数字识别
LeNet-5 项目说明
项目简介

本项目实现了经典的 LeNet-5 卷积神经网络模型，主要用于手写数字识别任务。模型结构包括两个卷积层、两个池化层和三个全连接层，适用于 MNIST 数据集。


>项目结构
.
├── model.py           # LeNet-5 模型定义
├── plot.py            # 数据加载与可视化
├── train.py           # 模型训练脚本
├── test.py            # 模型测试与可视化
├── best_model.pth     # 训练后的最佳模型权重
├── README.md          # 项目说明文档


安装依赖
>pip install torch torchvision matplotlib


## 数据加载与预处理

在 plot.py 中，定义了 test_Loader，用于加载 MNIST 测试数据集。数据预处理包括：

将图像转换为 Tensor

标准化图像数据

加载器使用 DataLoader 进行批处理


## 模型定义

在 model.py 中，定义了 LeNet-5 模型结构。模型包括以下层：


输入层：32x32 灰度图像

C1：卷积层，6 个 5x5 卷积核，输出 28x28 特征图

S2：池化层，2x2 平均池化，输出 14x14 特征图

C3：卷积层，16 个 5x5 卷积核，输出 10x10 特征图

S4：池化层，2x2 平均池化，输出 5x5 特征图

C5：卷积层，120 个 1x1 卷积核，输出 1x1 特征图

F6：全连接层，84 个神经元

输出层：10 个神经元，对应 10 个数字类别

## 模型训练

在 train.py 中，定义了模型训练过程，包括：

加载训练数据

定义损失函数和优化器

训练模型并保存最佳权重至 best_model.pth

模型测试与可视化

在 test.py 中，定义了模型测试过程：

加载测试数据

加载训练好的模型权重

计算测试准确率

可视化预测结果：

```python
import torch
import matplotlib.pyplot as plt
import model

def test_model_process(model, test_data, max_visualize=10):
    test_acc = 0.0
    test_num = 0
    visualize_count = 0  # 可视化计数
    model.eval()

    with torch.no_grad():
        for test_x, test_y in test_data:
            output = model(test_x)
            pre_label = torch.argmax(output, dim=1)
            test_acc += torch.sum(pre_label == test_y)
            test_num += test_x.size(0)

            # 遍历 batch
            for i in range(test_x.size(0)):
                if visualize_count >= max_visualize:
                    break

                label = test_y[i].item()
                result = pre_label[i].item()

                # 可视化
                img = test_x[i].squeeze().cpu()  # 去掉 channel
                plt.imshow(img, cmap='gray')
                title_color = 'green' if label == result else 'red'
                plt.title(f"预测值：{result} 真实值：{label}", color=title_color)
                plt.axis('off')
                plt.show()

                # 控制台输出
                if label == result:
                    print("预测值：", result, "-------", "真实值", label)
                else:
                    print("预测值：", result, "-----------------------", "真实值", label)

                visualize_count += 1

    test_avg_acc = test_acc.item() / test_num
    print("测试准确率:", test_avg_acc)
```

使用方法

训练模型：

>python train.py


测试模型并可视化：

>python test.py