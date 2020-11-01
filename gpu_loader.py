import torch

def get_default_device():
    """Pick GPU if available, else GPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensors to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.loader:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.loader)