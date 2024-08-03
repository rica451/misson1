# coding:UTF-8
import pickle
import re
import os
from sentiment.word_sequence import Word2Sequence

def save_vocab_to_pkl(vocab_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(vocab_dict, f)
def tokenize(text):
    fileters = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
                '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']
    text = re.sub("<.*?>", "", text, flags=re.S)
    text = re.sub("!".join(fileters), "", text, flags=re.S)
    return [i.strip() for i in text.split()]


if __name__ == '__main__':
    ws = Word2Sequence()
    path = "E:/diangpt/aclImdb_v1/aclImdb/train"
    temp_data_path = [os.path.join(path, 'pos'), os.path.join(path, 'neg')]
    for date_path in temp_data_path:
        file_paths = [os.path.join(date_path, file_name) for file_name in os.listdir(date_path)]
        for file_path in file_paths[1:1000000]:
            sentence = tokenize(open(file_path, encoding='utf-8').read())
            ws.fit(sentence)

    ws.build_vocab(min=10,max_features=10000)
    print(ws.dict)
    print(ws.count)
    print(len(ws))
    # 读取文本的训练集然后构建词汇表，保存为pkl的数据格式，方便后续的使用
    save_vocab_to_pkl(ws.dict, 'ws.pkl')




