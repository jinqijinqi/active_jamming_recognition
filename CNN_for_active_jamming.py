import numpy as np
import torch
from h5py import File
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# 超参数设置
debug = False       # 改成True进入调试模式
batch_size = 16
lr = 0.01
epochs = 15
train_ratio = 0.8

if debug:
    epochs = 1000
    batch_size = 2


def customDataset(stft_file, label_file):
    # data pre-processing
    echo_stft = File(stft_file, 'r')
    echo_stft = np.transpose(echo_stft['tf_data'])
    echo_stft = echo_stft.transpose([0, 3, 1, 2])

    label = File(label_file, 'r')
    label = np.transpose(label['gt_label'])
    label = label.transpose([1, 0])

    echo_stft = torch.tensor(echo_stft, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long).flatten()
    data = TensorDataset(echo_stft, label)
    train_size = int(len(data) * train_ratio)
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    return train_data, test_data


# Model build
class MyNet(nn.Module):
    def __init__(self) -> None:
        super(MyNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 13),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 128, 5),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(5632, 128),
            nn.ReLU(),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        feature = self.conv(x)
        feature = feature.view(x.shape[0], -1)
        output = self.fc(feature)
        return output


def train(dataloader, model, loss_fn, optimizer):
    model.train()

    if not debug:
        size = len(dataloader.dataset)
    else:
        size = batch_size * 1

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        if not debug:
            if batch % 10 == 0:
                print(f"loss: {loss:.6f}  [{current:5d}/{size:5d}]")
        else:
            print(f"train loss: {loss:.6f}")


def test(dataloader, model, loss_fn):
    model.eval()
    loss, correct = 0, 0
    num_batches = len(dataloader)

    if not debug:
        size = len(dataloader.dataset)
    else:
        size = batch_size * 1

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).float().sum().item()
        loss /= num_batches
        correct /= size
    print("Avg Loss: {:.6f}\tAccuracy: {:.2f}%".format(loss, correct*100))


def main():
    print("开始读取数据...")
    stft_file_path = "data/tf_data.mat"
    label_file_path = "data/gt_label.mat"
    train_data, test_data = customDataset(stft_file_path, label_file_path)
    print("读取数据完成!")

    print("开始编译模型...")
    model = MyNet()
    print("模型编译完成！")
    print(model)

    train_iter = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True, num_workers=8)
    test_iter = DataLoader(test_data, batch_size=batch_size,
                           shuffle=True, num_workers=8)

    if debug:
        print("调试模式（仅进行一个batch训练）：\n")
        train_iter = (next(iter(train_iter)))
        test_iter = (next(iter(test_iter)))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print("训练开始...")
        print("Epoch {}/{}:\n----------".format(epoch+1, epochs))
        train(train_iter, model, loss_fn, optimizer)
        test(test_iter, model, loss_fn)
        print("训练完成！")


if __name__ == "__main__":
    main()
