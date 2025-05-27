from early_stopping import EarlyStopping
from loss import custom_loss
import torch
import torch.nn.utils
from torch.optim.lr_scheduler import OneCycleLR

def train_model(model, train_loader, val_loader, num_epochs=250):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use the AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    # Use the OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.008,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
        anneal_strategy='cos'
    )
    
    # Early stop strategy
    early_stopping = EarlyStopping(patience=80, min_delta=1e-5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training stage
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = custom_loss(out, data.y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Verification stage
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                loss = custom_loss(out, data.y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Stop early for inspection
        early_stopping(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return train_losses, best_val_loss
