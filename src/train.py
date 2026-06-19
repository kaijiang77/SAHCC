import logging

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from src.data.crowd_datamodule import CrowdDataModule
from src.modules.lit_crowd import LitCrowdModel
from src.utils.seed import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    seed_everything(int(cfg.seed))
    logger.info('Config:\n%s', OmegaConf.to_yaml(cfg))

    datamodule = CrowdDataModule(cfg.data)
    model = LitCrowdModel(cfg)
    tb_logger = TensorBoardLogger(save_dir=cfg.paths.log_dir, name=cfg.run_name)

    ckpt_callback = ModelCheckpoint(
        dirpath=f"{cfg.paths.ckpt_dir}/{cfg.run_name}",
        filename='best-{best_mae:.2f}',
        monitor='best_mae',
        mode='min',
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.get('strategy', 'auto'),
        sync_batchnorm=cfg.trainer.get('sync_batchnorm', cfg.trainer.get('syn_batch', True)),
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        logger=tb_logger,
        callbacks=[ckpt_callback],
    )

    logger.info('Start training: run_name=%s', cfg.run_name)
    trainer.fit(model, datamodule=datamodule)
    logger.info('Training finished: run_name=%s', cfg.run_name)


if __name__ == '__main__':
    main()
