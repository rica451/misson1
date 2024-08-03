# coding:utf-8

import os
import torch
from torch.utils.data import Dataset, DataLoader
from sentiment.word_save import tokenize
from sentiment.word_sequence import Word2Sequence


path = "E:/diangpt/aclImdb_v1/aclImdb"
ws = Word2Sequence()
ws.load_vocab('ws.pkl')

class ImdbDataSet(Dataset):
    def __init__(self, mode):
        # 调用父类的方法初始化子类继承的属性
        super(ImdbDataSet, self).__init__()
        if mode == 'train':
            text_path = [os.path.join(path, 'train', 'neg'), os.path.join(path, 'train', 'pos')]
        else:
            text_path = [os.path.join(path, 'test', 'neg'), os.path.join(path, 'test', 'pos')]
        self.total_path = []
        for i in text_path:
            self.total_path += [os.path.join(i, j) for j in os.listdir(i)]
            # os.path.join(i, j) 是 Python 中 os.path 模块的一个方法，用于将多个路径组合成一个路径。

    def __getitem__(self, idx):
        # 读取文件
        cur_path = self.total_path[idx]
        cur_filename = os.path.basename(cur_path)
        label = int(cur_filename.split("_")[-1].split(".")[0]) - 1
        text = tokenize(open(cur_path, encoding='utf-8').read().strip())
        return label, text

    def __len__(self):
        return len(self.total_path)


def collate_fn(batch):
    # batch是list，其中是一个一元元组，每个元组是dataset中__getitem__的返回值
    label, content = list(zip(*batch))
    # label是元组，content是列表
    content = [ws.transform(i,max_len=100) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    # print("content", content)
    return label, content


def get_dataloader(train_data=True):
    mode = ""
    if train_data:
        mode = "train"
    imdb_dateset = ImdbDataSet(mode)
    dataloader = DataLoader(
        dataset=imdb_dateset,
        batch_size=3,
        shuffle=False,
        collate_fn=collate_fn
    )
    return dataloader


if __name__ == '__main__':
    dataset = ImdbDataSet(mode='train')
    print(dataset[0])
    dataloader = get_dataloader()
    for idx, (label, text) in enumerate(dataloader):
        print("idx", idx)
        print("label", label)
        print("text", text)
        break
