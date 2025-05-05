import os


def remove_checkpoints(trainer, checkpoint_path=None):
    """Remove existing checkpoints from the logger directory.

    Args:
        trainer (pytorch_lightning.Trainer): The trainer object.
        checkpoint_path (str, optional): The path to the checkpoint directory.
            Defaults to None.
    """
    # Get the checkpoint directory path from the logger
    if checkpoint_path is None:
        logger = trainer.logger
        checkpoint_path = os.path.join(
            logger.save_dir,
            str(logger.name),
            "version_" + str(logger.version),
            "checkpoints",
        )

        # Remove existing checkpoints if the directory exists
        if os.path.exists(checkpoint_path):
            for filename in os.listdir(checkpoint_path):
                file_path = os.path.join(checkpoint_path, filename)
                os.remove(file_path)
