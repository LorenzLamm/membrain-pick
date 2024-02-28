import os 
import torch

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