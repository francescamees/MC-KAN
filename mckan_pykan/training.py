try:
    from pytorch_lightning import seed_everything
except Exception:  # pragma: no cover - fallback for older/newer PL layouts
    from lightning_fabric.utilities.seed import seed_everything
import pytorch_lightning as pl
from .lit_module import LitMCKANPyKAN
from .data import create_train_val_test_dataloaders


def training(seed, config, dataset):
    seed_everything(seed=seed, workers=True)
    train_dataloader, val_dataloader, _ = create_train_val_test_dataloaders(config, dataset)
    litmodel = LitMCKANPyKAN(config)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir='logs/mckan_pykan', name='mckan_pykan')

    best_val_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best_val'
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=False,
        mode='min'
    )

    trainer = pl.Trainer(
        deterministic=True,
        devices=1,
        enable_model_summary=False,
        enable_progress_bar=False,
        accelerator='cpu',
        max_epochs=config.num_epochs,
        logger=tb_logger,
        check_val_every_n_epoch=10,
        log_every_n_steps=1,
        callbacks=[best_val_checkpoint, early_stop_callback],
    )

    trainer.fit(
        model=litmodel,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return litmodel
