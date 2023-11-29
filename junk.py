from torch.utils.data import DataLoader

from src.dataset import SlidingWindowIMUsDataset

def test():
    data_dir = '../data/train'
    train_dataset = SlidingWindowIMUsDataset(data_dir=data_dir, window_len=20000, hop=500, sample_len=2000, augmentation=False, ambidextrous=False)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)


    for iteration, (data, labels) in enumerate(train_dataloader):
        if iteration == 0:
            print(data.shape)
            print(labels.shape)
            print(labels)
            break

if __name__ == '__main__':
    test()

