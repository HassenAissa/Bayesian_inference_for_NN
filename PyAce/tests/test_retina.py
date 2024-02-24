from medmnist import RetinaMNIST

train_dataset = RetinaMNIST(split="train", download = True)
print(train_dataset)

