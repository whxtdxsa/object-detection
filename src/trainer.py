import torch
from tqdm import tqdm

def train_one_epoch(network, dataloader, optimizer, criterion, device, amp_context=None, scaler=None):
    network.train()
    total_loss = 0
    
    for x_batch, t_batch in tqdm(dataloader, desc="Training", leave=False):
        # x_batch, t_batch = x_batch.to(device), t_batch.to(device)
        x_batch = x_batch.to(device)

        optimizer.zero_grad()
        with amp_context:
            y_batch = network(x_batch)
            loss = criterion(y_batch, t_batch)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_loss(network, dataloader, criterion, device, amp_context=None):
    network.eval()
    total_loss = 0

    with torch.no_grad():
        for x_batch, t_batch in dataloader:
            # x_batch, t_batch = x_batch.to(device), t_batch.to(device)
            x_batch = x_batch.to(device)

            with amp_context:
                y_batch = network(x_batch)
                loss = criterion(y_batch, t_batch)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_accuracy(network, dataloader, device, amp_context=None):
    network.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, t_batch in dataloader:
            # x_batch, t_batch = x_batch.to(device), t_batch.to(device)
            x_batch = x_batch.to(device)

            with amp_context:
                y_batch = network(x_batch)
            
            y_batch = y_batch.argmax(dim=1)
            correct += (y_batch == t_batch).sum().item()
            total += t_batch.size(0)

    return correct / total


def save_checkpoint(model, path="checkpoint.pt"):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path="checkpoint.pt", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))



