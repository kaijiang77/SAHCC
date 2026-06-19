import hydra
import lightning as L
from omegaconf import DictConfig

from src.data.crowd_datamodule import CrowdDataModule
from src.modules.lit_crowd import LitCrowdModel


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    if not cfg.eval.ckpt_path:
        raise ValueError('Please provide eval.ckpt_path')

    datamodule = CrowdDataModule(cfg.data)
    model = LitCrowdModel.load_from_checkpoint(cfg.eval.ckpt_path, cfg=cfg)
    trainer = L.Trainer(accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices)
    trainer.validate(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
