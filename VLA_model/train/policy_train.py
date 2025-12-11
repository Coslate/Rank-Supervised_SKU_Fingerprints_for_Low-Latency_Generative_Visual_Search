import torch
from torch.utils.data import DataLoader
from model.policy import VLAPolicy
from data.data_retrieve import VLADataset
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm


def train_policy(samples, visual_dim, quality_dim, num_actions, device="cuda", num_epochs=10000):
    dataset = VLADataset(samples)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = VLAPolicy(visual_dim, quality_dim, num_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        total, correct, loss_sum = 0, 0, 0

        # Progress bar for batches within each epoch
        batch_pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, unit="batch")
        
        for vfeat, qfeat, labels in batch_pbar:
            vfeat, qfeat, labels = (
                vfeat.to(device),
                qfeat.to(device),
                labels.to(device),
            )

            logits = model(vfeat, qfeat)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item() * labels.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # Update batch progress bar with current metrics
            current_loss = loss_sum / total if total > 0 else 0.0
            current_acc = correct / total if total > 0 else 0.0
            batch_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.3f}'
            })

        # Calculate epoch metrics
        epoch_loss = loss_sum / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'acc': f'{epoch_acc:.3f}'
        })
        
        # Print detailed info every 100 epochs or on first epoch
        if (epoch + 1) % 100 == 0 or epoch == 0:
            tqdm.write(
                f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.3f}"
            )

    epoch_pbar.close()
    return model
