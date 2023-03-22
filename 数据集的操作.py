from datasets import load_dataset

dataset = load_dataset(path='seamew/ChnSentiCorp', split = 'train')

# print(dataset[0])

# sort 排序
# 未排序的label是乱的
print(dataset['label'][:10]) #[1, 1, 0, 0, 1, 0, 0, 0, 1, 1]

# 排序之后label就是有序的
sorted_dataset = dataset.sort('label')
print(sorted_dataset['label'][:10])
print(sorted_dataset['label'][-10:])
