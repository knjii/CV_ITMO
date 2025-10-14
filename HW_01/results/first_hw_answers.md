## Вариант 2
### Архитектура модели
```
class HomeworkNet_0(nn.Module):
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
        # print("After conv1:", x.shape)
        # print("Среднее значение активаций:", x.mean().item())
        
        # conv2
        x = self.conv2(x)
        x = F.relu(self.bn_conv2(x))
        # print("After conv2:", x.shape)
        # print("Среднее значение активаций:", x.mean().item())

        # conv3
        x = self.conv3(x)
        x = F.relu(self.bn_conv3(x))
        # print("After conv3:", x.shape)
        # print("Среднее значение активаций:", x.mean().item())

        # gap + fc1 
        x = self.drop(x)
        x = self.gap(x)
        # print("After GAP:", x.shape)
        # print("Среднее значение активаций:", x.mean().item())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
```
### Результаты экспериментов
 #### Сравните выход с использованием padding=0 и padding=1, оцените разницу в размерности и среднее значение активаций:
 1. Первая сеть: 
 Размерность после conv1: torch.Size([1, 16, 24, 24]) Среднее значение активаций: 0.39624857902526855 
 Размерность после conv2: torch.Size([1, 32, 20, 20]) Среднее значение активаций: 0.4000762104988098 
 Размерность после conv3: torch.Size([1, 64, 16, 16]) Среднее значение активаций: 0.3992786109447479 
 Размерность после GAP: torch.Size([1, 64, 1, 1]) Среднее значение активаций: 0.40319645404815674 

 2. Вторая сеть: 
 Размерность после conv1: torch.Size([1, 16, 26, 26]) Среднее значение активаций: 0.3983019292354584 
 Размерность после conv2: torch.Size([1, 32, 22, 22]) Среднее значение активаций: 0.3987589180469513 
 Размерность после conv3: torch.Size([1, 64, 18, 18]) Среднее значение активаций: 0.3987939655780792 
 Размерность после GAP: torch.Size([1, 64, 1, 1]) Среднее значение активаций: 0.3996787965297699 
 
 #### Вывод: размерность выхода при padding=0 будет на два меньше; средние значения активаций не сильно изменяются по двум возможным причинам: 

    1. Padding занулялся на первом слое. В силу особенности датасета CIFAR, изображения на картинках чаще всего "центрированы", следствием чего является отсутствие ключевых паттернов по краям картинок.

    2. Я использовал BatchNorm перед функциями активации, что центрирует и масштабирует предактивации


## Вариант 3
### Архитектура:
 ```
class HomeworkNet_compact(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=64,
            kernel_size=3,
            padding=0 # 16 -> 14
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=8,
            kernel_size=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.bn_conv1 = nn.BatchNorm2d(num_features=16)
        self.bn_conv2 = nn.BatchNorm2d(num_features=64)
        self.bn_conv3 = nn.BatchNorm2d(num_features=8)
        self.bn_conv4 = nn.BatchNorm2d(num_features=16)
        self.mxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=3136, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mxp1(x) # 32 -> 16 
        x = F.relu(self.bn_conv1(x))

        x = self.conv2(x)
        x = F.relu(self.bn_conv2(x))

        x = self.conv3(x)
        x = F.relu(self.bn_conv3(x))

        x = self.conv4(x)
        x = F.relu(self.bn_conv4(x))

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
 ```

 **Проверка лимита количества параметров:**
 ```
assert sum([param.numel() for param in model2.parameters()]) <= 50000, "Количество параметров вышло за рамки установленного лимита"
 ```

 **Количество параметров**
 ```
 42994
 ```