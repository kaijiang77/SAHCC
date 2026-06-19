import lightning as L
from torch.utils.data import DataLoader

from src.data.collate import collate_fn_crowd
from src.data.unified_dataset import build_unified_datasets


class CrowdDataModule(L.LightningDataModule):
    def __init__(self, cfg_data):
        super().__init__()
        self.cfg = cfg_data
        self.train_set = None
        self.val_set = None
        self.train_eval_set = None

    def setup(self, stage=None):
        self.train_set, self.val_set, self.train_eval_set = build_unified_datasets(self.cfg)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn_crowd,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            drop_last=False,
            collate_fn=collate_fn_crowd,
            num_workers=self.cfg.num_workers,
        )
