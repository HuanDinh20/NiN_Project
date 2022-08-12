import torch
from module import NiN
import utils
from train import per_epoch_activity
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 1. initialize
batch_size = 16
EPOCH = 15
timestamps = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = fr"A:\huan_shit\Study_Shit\Deep_Learning\Side_Projects\NiN\saved_model\full_model\Nin_model_01.pth"
root = r'A:\huan_shit\Study_Shit\Deep_Learning\Side_Projects\LeNet_project\data'
device = utils.get_device()
summary_writer = SummaryWriter()
# 2.  dataset
train_aug, test_aug = utils.image_augmentation()
train_set, test_set = utils.get_FashionMNIST(train_aug, test_aug, root)
train_loader, val_loader = utils.create_dataloader(train_set, test_set, batch_size)

#3. Model
model = NiN()
model.to(device)
# 4,5. loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


if __name__=="__main__":
    per_epoch_activity(train_loader, val_loader, device, optimizer, model, loss_fn, summary_writer,
                       EPOCH, timestamps)
    torch.save(model, model_path)


