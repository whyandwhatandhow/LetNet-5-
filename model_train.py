import copy
import time
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import plot
import model
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_val_data_process():
    train_set = plot.trainSet
    train_len = int(0.8 * len(train_set))
    val_len = len(train_set) - train_len
    train_data, val_data = random_split(train_set, [train_len, val_len])

    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=True, num_workers=0)

    return train_loader, val_loader


def train_model_process(model, train_dataloader, val_dataloader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝出参数们

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()  # 保存当前时间

    for epoch in range(epochs):
        print("Epoch{}/{}".format(epoch + 1, epochs))
        print("-" * 10)

        train_loss = 0.0
        train_acc = 0

        val_loss = 0.0
        val_acc = 0

        train_num = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            model.train()  # 训练模式
            optimizer.zero_grad()
            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)

            loss = criterion(output, b_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_acc += torch.sum(pre_label == b_y).item()
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            model.eval()
            with torch.no_grad():
                output = model(b_x)
                pre_label = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_acc += torch.sum(pre_label == b_y).item()
                val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_acc / train_num)
        val_acc_all.append(val_acc / val_num)

        print("Epoch:{:} Train Loss:{:.4f}  ACC:{:.4f}".format(epoch, train_loss / train_num, train_acc / train_num))
        print("Epoch:{:} Val Loss:{:.4f}  ACC:{:.4f}".format(epoch, val_loss / val_num, val_acc / val_num))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "best_model.pth")

    model.load_state_dict(best_model_wts)
    time_use = time.time() - since

    train_process = pd.DataFrame({
        "epoch": list(range(1, epochs + 1)),
        "train_loss_all": train_loss_all,
        "train_acc_all": train_acc_all,
        "val_loss_all": val_loss_all,
        "val_acc_all": val_acc_all
    })
    train_process["time_use"] = time_use
    train_process["best_acc"] = best_acc

    return train_process


def matlib_acc_loss(train_process):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,"bo-",label="Train_Loss")
    plt.plot(train_process["epoch"],train_process.val_loss_all,"ro-",label="Val_Loss")
    plt.title("Train_Val_Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'bo-', label='Train Acc')
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'ro-', label='Val Acc')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


model = model.LeNet()
train_data, val_data = train_val_data_process()
process = train_model_process(model, train_data, val_data, 100)
matlib_acc_loss(process)
print(process)
