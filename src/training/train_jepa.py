import torch
import torch.optim as optim
import torch.nn.functional as F
from config.config import get_config
import torch
import torch.optim as optim
from utils.early_stop import get_lr_schedule, EarlyStopping
from src.models.jepa import MiniJEPA
from src.data.data_processing import get_data


def train():

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    best_val_loss = float('inf')

    CONFIG = get_config()

    train_loader, test_loader = get_data()

        # Initialize model
    print("\n" + "=" * 60)
    print("Initializing Model")
    print("=" * 60)

    model = MiniJEPA(CONFIG).to(CONFIG['device'])
    print(f"✓ Model parameters: {model.count_parameters():,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['lr'], 
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = get_lr_schedule(
        optimizer, 
        CONFIG['warmup_epochs'], 
        CONFIG['epochs']
    )
    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(patience=CONFIG['patience'])

    print("\n" + "=" * 60)
    print(f"Starting Training on {CONFIG['device']}")
    print("=" * 60)

    for epoch in range(CONFIG["epochs"]):
        # ============ TRAINING ============
        model.train()
        total_loss = 0.0
        num_batches = 0

        for i, (images, context_mask, target_masks) in enumerate(train_loader):
            images = images.to(CONFIG["device"], non_blocking=True)
            context_mask = context_mask.to(CONFIG["device"], non_blocking=True)
            target_masks = [m.to(CONFIG["device"], non_blocking=True) for m in target_masks]

            optimizer.zero_grad(set_to_none=True)

            try:
                with torch.amp.autocast():
                    # Forward pass
                    with torch.no_grad():
                        target_output = model.forward_target(images)
                    
                    context_output = model.forward_context(images, context_mask)
                    predictions = model.forward_predictor(context_output, target_masks)

                    # Compute loss
                    loss = 0.0
                    valid_blocks = 0
                    for pred, t_mask in zip(predictions, target_masks):
                        n_masked = t_mask.sum()
                        if n_masked == 0:
                            continue
                        loss += F.mse_loss(pred[t_mask], target_output[t_mask])
                        valid_blocks += 1

                    if valid_blocks == 0:
                        continue  # Skip this batch
                    
                    loss = loss / valid_blocks

                # Check for NaN/Inf
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss detected at epoch {epoch+1}, batch {i}")
                    continue

                # Backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                # EMA Update for target encoder
                with torch.no_grad():
                    m = CONFIG['ema_momentum']
                    for param_q, param_k in zip(
                        model.context_encoder.parameters(), 
                        model.target_encoder.parameters()
                    ):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.data)

                total_loss += loss.item()
                num_batches += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at epoch {epoch+1}, batch {i}. Skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

        # Update learning rate
        scheduler.step()
        
        avg_train_loss = total_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        # ============ VALIDATION ============
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, context_mask, target_masks in test_loader:
                images = images.to(CONFIG["device"], non_blocking=True)
                context_mask = context_mask.to(CONFIG["device"], non_blocking=True)
                target_masks = [m.to(CONFIG["device"], non_blocking=True) for m in target_masks]

                try:
                    target_output = model.forward_target(images)
                    context_output = model.forward_context(images, context_mask)
                    predictions = model.forward_predictor(context_output, target_masks)

                    v_loss = 0.0
                    v_valid = 0
                    for pred, t_mask in zip(predictions, target_masks):
                        if t_mask.sum() == 0:
                            continue
                        v_loss += F.mse_loss(pred[t_mask], target_output[t_mask])
                        v_valid += 1
                    
                    if v_valid > 0:
                        val_loss += (v_loss / v_valid).item()
                        val_batches += 1
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

        avg_val_loss = val_loss / max(val_batches, 1)

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(current_lr)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': CONFIG
            }, "cosmic_jepa_best.pth")
            print(f"✓ Best model saved!")

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'config': CONFIG,
            'history': history
        }, "cosmic_jepa_latest.pth")

        # Print progress
        print(f"Epoch [{epoch+1:3d}/{CONFIG['epochs']}] | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Best: {best_val_loss:.4f}")

        # Early stopping check
        # early_stopping(avg_val_loss)
        # if early_stopping.early_stop:
        #     print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
        #     break

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved: cosmic_jepa_best.pth, cosmic_jepa_latest.pth")

    return model, history

