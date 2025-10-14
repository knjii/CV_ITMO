### Первая архитектура:
```
class HometaskNet_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=9, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=1)
        self.fc1 = nn.Linear(in_features=16384, out_features=10) # in_features = 64*16*16
        # ReLU используем как F.ReLU
        # softmax зашита в CrossEntropyLoss

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
```

**Проблема**: модель переобучена. Предположительно, причина в том, что у нас много обучающихся параметров, при этом отсутствует регуляризация.
==>
Необходимо добавить слой Dropout и BatchNorm, установить параметр weight_decay в Adam

### Вторая архитектура:
```
class HometaskNet_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=9, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=1)
        self.fc1 = nn.Linear(in_features=16384, out_features=10) # in_features = 64*16*16
        self.drop = nn.Dropout(p=0.3)
        self.bn_conv1 = nn.BatchNorm2d(num_features=16)
        self.bn_conv2 = nn.BatchNorm2d(num_features=32)
        self.bn_conv3 = nn.BatchNorm2d(num_features=64)
        # ReLU используем как F.ReLU
        # softmax зашита в CrossEntropyLoss

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = F.relu(self.bn_conv1(x))
        # conv2
        x = self.conv2(x)
        x = F.relu(self.bn_conv2(x))
        # conv3
        x = self.conv3(x)
        x = F.relu(self.bn_conv3(x))
        # fc1
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc1(x)
        return x
```
![alt text](image-1.png)

#### Метрики:
Accuracy: 0.6514
F1 (macro): 0.6554
F1 (weighted): 0.6554

**Проблема**: нейросеть все еще немного переобучается, после 9-12 эпохи лосс на валидационной выборке стабилизируется, в то время как лосс на тренировочной выборке продолжает падать.
==>
Стоит попробовать использовать GAP вместо полносвязного слоя, что часто помогает при переобучении, также сделает модель легче

### Третья архитектура:
```
class HometaskNet_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=9, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding=1)
        self.fc1 = nn.Linear(in_features=64, out_features=10) # in_features = 64*1*1 (так как используем GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.3)
        self.bn_conv1 = nn.BatchNorm2d(num_features=16)
        self.bn_conv2 = nn.BatchNorm2d(num_features=32)
        self.bn_conv3 = nn.BatchNorm2d(num_features=64)
        # ReLU используем как F.ReLU
        # softmax зашита в CrossEntropyLoss

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        x = F.relu(self.bn_conv1(x))
        # conv2
        x = self.conv2(x)
        x = F.relu(self.bn_conv2(x))
        # conv3
        x = self.conv3(x)
        x = F.relu(self.bn_conv3(x))
        # gap + fc1 
        x = self.drop(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
```
![alt text](image.png)

#### Метрики:
Accuracy: 0.6521
F1 (macro): 0.6411
F1 (weighted): 0.6411

