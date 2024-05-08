import os 
import torch
from torch.nn import MSELoss


class CombinedLoss(torch.nn.Module):
    def __init__(self, criteria):
        super().__init__()
        self.criteria = criteria
    def forward(self, preds, targets, weights):
        loss = 0.
        for key in self.criteria:
            loss += self.criteria[key](preds[key], targets[key], weights[key])
        return loss

class weighted_MSELoss(MSELoss):
    def __init__(self):
        super(weighted_MSELoss, self).__init__()
        self.mse = MSELoss(reduction="sum")

    def forward(self, input, target, weights):
        mse = self.mse(weights.squeeze() * input.squeeze(), weights.squeeze() * target.squeeze())
        mse /= weights.sum()
        return mse


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir='checkpoints', filename='checkpoint'):
    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_config': model.get_config()
    }
    
    # Define the checkpoint file path
    checkpoint_file_path = os.path.join(checkpoint_dir, f'{filename}_epoch_{epoch}.pth')
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_file_path)
    print(f'Checkpoint saved: {checkpoint_file_path}')
    return checkpoint_file_path