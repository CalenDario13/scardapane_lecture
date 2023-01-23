import torch, torchaudio
from torch import nn
from torch.nn import functional as F
print(torch.cuda.is_available())

import pytorch_lightning as pl
from torchmetrics import functional

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

DATAPATH = Path('G:\My Drive\DS\doker_kuberneast\scardapane_lectures\data\ESC-50-master\ESC-50-master')

class ESC50Dataset(torch.utils.data.Dataset):
    # Simple class to load the desired folders inside ESC-50
    
    def __init__(self, path: Path, 
                 sample_rate: int = 8000,
                 folds: list  = [1]):
        # Load CSV & initialize all torchaudio.transforms:
        # Resample --> MelSpectrogram --> AmplitudeToDB
        self.path = path
        self.csv = pd.read_csv(path / Path('meta/esc50.csv'))
        self.csv = self.csv[self.csv['fold'].isin(folds)]
        self.resample = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=sample_rate
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        
        
    def __getitem__(self, index):
        # Returns (xb, yb) pair, after applying all transformations on the audio file.
        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(self.path / 'audio' / row['filename'])
        label = row['target']
        xb = self.db(
            self.melspec(
                self.resample(wav)
            )
        )
        return xb, label
        
    def __len__(self):
        # Returns length
        return len(self.csv)
    
class AudioNet(pl.LightningModule):
    
    def __init__(self, n_classes = 50, base_filters = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, base_filters, 11, padding=5)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(base_filters, base_filters * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters * 2)
        self.conv4 = nn.Conv2d(base_filters * 2, base_filters * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters * 4)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(base_filters * 4, n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x[:, :, 0, 0])
        return x
    
    def training_step(self, batch, batch_idx):
        # Very simple training loop
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = functional.accuracy(y_hat, y, task="multiclass", num_classes=50)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return acc
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=1)
        acc = functional.accuracy(y_hat, y, task="multiclass", num_classes=50)
        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train(datapath, accelerator: str = 'gpu', max_epoch: int = 10):

    # We use folds 1,2,3 for training, 4 for validation, 5 for testing.
    train_data = ESC50Dataset(datapath, folds=[1,2,3])
    val_data = ESC50Dataset(datapath, folds=[4])
    test_data = ESC50Dataset(datapath, folds=[5])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=2)
    # test_loader = torch.utils.data.DataLoader(test_data, num_workers=6, batch_size=4)
    
    pl.seed_everything(0)
    
    audionet = AudioNet()
    trainer = pl.Trainer(accelerator=accelerator, max_epochs=max_epoch)
    trainer.fit(audionet, train_loader, val_loader)

if __name__ == "main":
    
    train(DATAPATH)






