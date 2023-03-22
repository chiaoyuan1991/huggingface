# coding = utf-8
from transformers import BertTokenizer

# 加载预训练字典和分词方法

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

sents = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小，其他的都一般',
    '今天才知道这书还有第6卷，真有点郁闷。',
    '机器背面似乎被撕了张什么标签，残胶还在。',
]

# tokenizer, sents

out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],

    # 当句子长度大于max_length时，截断
    truncation=True,

    # 一律补pad到max_length长度
    padding='max_length',
    add_special_tokens=True,
    max_length=30,
    return_tensors=None,
)

print(out)

tokenizer.decode(out)
print(tokenizer.decode(out))
