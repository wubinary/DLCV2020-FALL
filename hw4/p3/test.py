from dataset import MiniImageNet_Dataset, get_transform

trans,_ = get_transform()
train_set = MiniImageNet_Dataset('../hw4_data/train/',trans)
valid_set = MiniImageNet_Dataset('../hw4_data/val/',trans)

print(train_set.data[:10],train_set.label[:10])
print(valid_set.data[:10],valid_set.label[:10])
print(len(set(train_set.label)))

