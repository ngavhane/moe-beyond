import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter
import random
from collections import OrderedDict

# ==== Global Constants ====
NUM_LAYERS = 27
LAYER_EMBED_DIM = 512
NUM_EXPERTS = 64
TOKEN_EMBED_DIM = 2048
INPUT_DIM = TOKEN_EMBED_DIM + LAYER_EMBED_DIM
TRANSFORMER_DIM = 512
MAX_LENGTH = 2048
SEED = 42
EARLY_STOP_PATIENCE = 5
CACHE_CAPACITY = 1000
# ===========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ExpertActivationDataset(Dataset):
    def __init__(self, data_dir, max_length=MAX_LENGTH, max_files=None):
        self.data_dir = data_dir
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.csv_files = all_files[:max_files] if max_files else all_files
        self.max_length = max_length
        self.layer_embedding = nn.Embedding(NUM_LAYERS, LAYER_EMBED_DIM)
        self.cache = OrderedDict()
        
    def __len__(self):
        return len(self.csv_files)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        
        if len(self.cache) >= CACHE_CAPACITY:
            self.cache.popitem(last=False)
            
        file_path = os.path.join(self.data_dir, self.csv_files[idx])
        df = pd.read_csv(file_path)

        required_columns = ['Token Embedding Vector', 'Layer ID', 'Token', 'Activated Expert IDs']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in file: {file_path}")

        token_embeddings = np.stack(
            df['Token Embedding Vector'].apply(
                lambda x: np.array(x.strip('[]').split(','), dtype=np.float32)
            ).values
        )

        layer_ids = torch.LongTensor(df['Layer ID'].values)
        #tokens = torch.LongTensor(df['Token'].values)

        def create_multi_hot(expert_ids):
            experts = list(map(int, expert_ids.strip('[]').split(',')))
            return torch.sum(torch.nn.functional.one_hot(
                torch.tensor(experts), num_classes=NUM_EXPERTS), dim=0).float()

        targets = torch.stack(df['Activated Expert IDs'].apply(create_multi_hot).tolist())
        layer_embeds = self.layer_embedding(layer_ids)
        token_embeddings = torch.FloatTensor(token_embeddings)

        combined_features = torch.cat([token_embeddings, layer_embeds], dim=-1)
        
        item = {
            'features': combined_features,
            'targets': targets,
            'length': len(df)
        }
        self.cache[idx] = item
        return item

def collate_fn(batch):
    features = [item['features'] for item in batch]
    targets = [item['targets'] for item in batch]
    lengths = [item['length'] for item in batch]

    padded_features = pad_sequence(features, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True)
    
    # Create attention mask (1 for real data, 0 for padding)
    mask = (padded_features.sum(dim=-1) != 0).float()
    
    return {
        'features': padded_features,
        'targets': padded_targets,
        'lengths': lengths,
        'mask': mask
    }

class ExpertPredictor(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, d_model=TRANSFORMER_DIM, 
                 nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, NUM_EXPERTS)
        )

    def forward(self, x, mask):
        x = self.input_proj(x)
        x = self.transformer_encoder(x, src_key_padding_mask=~mask.bool())
        return self.output_layer(x)

def train(model, dataloader, optimizer, device, writer, epoch, scaler):
    model.train()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    for batch_idx, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()
        
        with autocast(device_type='cuda'):
            outputs = model(features, mask)
            loss = (criterion(outputs, targets) * mask.unsqueeze(-1)).sum()
            valid_positions = mask.sum() * NUM_EXPERTS
            loss /= valid_positions

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            masked_preds = preds * mask.unsqueeze(-1)
            masked_targets = targets * mask.unsqueeze(-1)
            
            acc = accuracy_score(
                masked_targets.cpu().flatten().numpy(),
                masked_preds.cpu().flatten().numpy()
            )
            f1 = f1_score(
                masked_targets.cpu().flatten().numpy(),
                masked_preds.cpu().flatten().numpy(),
                average='macro'
            )

        step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('train/loss', loss.item(), step)
        writer.add_scalar('train/accuracy', acc, step)
        writer.add_scalar('train/f1_score', f1, step)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss.item():.4f} Acc: {acc:.4f} F1: {f1:.4f}')

def validate(model, dataloader, device, writer, epoch):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    total_loss = 0
    all_preds = []
    all_targets = []
    valid_positions = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['mask'].to(device)

            outputs = model(features, mask)
            loss = (criterion(outputs, targets) * mask.unsqueeze(-1)).sum()
            total_loss += loss.item()
            valid_positions += mask.sum() * NUM_EXPERTS

            preds = (torch.sigmoid(outputs) > 0.5).float()
            masked_preds = preds * mask.unsqueeze(-1)
            masked_targets = targets * mask.unsqueeze(-1)
            
            all_preds.append(masked_preds.cpu())
            all_targets.append(masked_targets.cpu())

    avg_loss = total_loss / valid_positions
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()
    
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')

    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/accuracy', acc, epoch)
    writer.add_scalar('val/f1_score', f1, epoch)

    print(f'\nValidation set: Avg loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}\n')
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser(description='Expert Activation Predictor Training')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--checkpoint-path', type=str, default='./logs/best_model.pth')
    parser.add_argument('--max-files', type=int, default=None, help='Maximum number of CSV files to use (for exploration)')

    # New hyperparameters
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of transformer encoder layers')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads in transformer')
    parser.add_argument('--dim-feedforward', type=int, default=2048,
                        help='Feedforward dimension in transformer')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                        help='Dropout rate throughout model')
    
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset and DataLoaders
    dataset = ExpertActivationDataset(args.data_dir, max_files=args.max_files)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           collate_fn=collate_fn, pin_memory=True)

    # Model and Optimizer
    model = ExpertPredictor(
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout_rate
    ).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.input_proj.parameters(), 'lr': args.lr},
        {'params': model.transformer_encoder.parameters(), 'lr': args.lr*0.9},
        {'params': model.output_layer.parameters(), 'lr': args.lr*0.8}
    ], weight_decay=0.01)
    
    scaler = GradScaler('cuda')
    writer = SummaryWriter(log_dir=args.log_dir)

    # Checkpoint Loading
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1

    if args.eval_only:
        validate(model, val_loader, device, writer, 0)
        return

    # Training Loop
    early_stop_counter = 0
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        train(model, train_loader, optimizer, device, writer, epoch, scaler)
        val_loss, val_acc = validate(model, val_loader, device, writer, epoch)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'scaler_state_dict': scaler.state_dict()
        }, os.path.join(args.log_dir, f'checkpoint_epoch_{epoch}.pth'))

        # Update best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOP_PATIENCE} epochs without improvement.")
            break

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch} completed in {epoch_time//60:.0f}m {epoch_time%60:.0f}s')

    writer.close()

if __name__ == '__main__':
    main()
