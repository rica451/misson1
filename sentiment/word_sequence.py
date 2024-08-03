import pickle


class Word2Sequence:
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    UNK = 0
    PAD = 1

    def __init__(self):
        self.inverse_dict = None
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}  # 词频的统计

    def fit(self, sentence):
        """
        把单个的句子保存到词频字典中
        :param sentence:文本中的句子
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_features=None):
        """
        根据条件构建词典
        :param min: 最小词频
        :param max: 最大词频
        :param max_features: 最大词数
        :return:
        """
        if min is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min}
        if max is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max}
        if max_features is not None:
            self.count = dict(sorted(self.count.items(), key=lambda x: x[1], reverse=True)[:max_features])
        # 排序的依据是单词的频率，频率高的在前面
        for word in self.count:
            self.dict[word] = len(self.dict)
        # 反转字典
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        把句子转换为数字序列, 单词-编号
        神经网络模型的输入必须是固定长度，所以需要对句子进行截断或者补全
        :param sentence:
        :param max_len:
        :return:
        """
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            else:
                sentence = sentence[:max_len]
        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        """
        把数字序列转换为单词
        :param indices: [1,2,3,4]
        :return: ['word1', 'word2', 'word3', 'word4']
        """
        return [self.inverse_dict.get(idx) for idx in indices]

    def load_vocab(self, file_path):
        with open(file_path, 'rb') as f:
            self.dict = pickle.load(f)
        self.inverse_dict = {v: k for k, v in self.dict.items()}

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    word2sequence = Word2Sequence()
    word2sequence.fit(['a', 'b', 'a', 'c'])
    word2sequence.fit(['a', 'b', 'a', 'c'])
    word2sequence.fit(['a', 'b', 'a', 'c'])
    word2sequence.build_vocab(min=0)
    print(word2sequence.dict)
    print(word2sequence.count)
    print(word2sequence.transform(['a', 'b', 'b', 'c', 'd', 'e', 'f'], max_len=10))
    print(word2sequence.inverse_transform([1, 2, 3, 4]))
