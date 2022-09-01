import pandas as pd
import torch
import PIL.Image
from torchvision import transforms
import  numpy as np
import  torch
import torchvision
import torch.nn.functional as F
from torch import  nn
from torch import optim

data_file = "data/lungdatasets.csv"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)
# csv使用分号分割，seq指定分隔符为；
data = pd.read_csv(data_file,sep=';')

# 住院时长周数为值
outputs = data.iloc[:, 25]

# 定义特征
all_features = data.iloc[:, 1:25]
print(all_features.dtypes)
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'int64'].index

# print(numeric_features[1])
# print(all_features[numeric_features])

all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)


# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)



train_batch_size = 64
test_batch_size = 256
corona_size = 24

def get_dataloader(train=True):
    assert isinstance(train, bool),"train必须为bool类型"
    #assert类似trycatch？

    dataset = all_features
    #准备数据迭代器
    batch_size = test_batch_size if train else test_batch_size
    #a = b if c else d 即如果c为Ture 则 a=b 否则 a = del
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return dataloader

print(all_features)
