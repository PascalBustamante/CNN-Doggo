import torch


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """
    Save model checkpoint.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss
