from torch.utils.data import Dataset
import torch

class VLADataset(Dataset):
    def __init__(self, samples):
        """
        samples is a list of:
        {
            "visual": np.array,
            "quality": np.array,
            "label": int
        }
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        return (
            torch.tensor(s["visual"], dtype=torch.float32),
            torch.tensor(s["quality"], dtype=torch.float32),
            torch.tensor(s["label"], dtype=torch.long),
        )
