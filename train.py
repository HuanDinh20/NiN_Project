import torch


def train_one_epoch(train_loader, device, optimizer, model, loss_fn, summary_writer, epoch_idx):
    """
    1. get batch of data, move to device
    2. zeros gradients
    3. perform inference
    4. calculate loss
    5. backward
    6. optimizer step
    7. report every 1000 batch
    """
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # 1.
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # 2. zeros grad
        optimizer.zero_grad()
        # 3. perform inference + compute loss
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # 4. backward
        loss.backward()
        # 5. tell optimizer step, adjust weights
        optimizer.step()

        running_loss += loss.item()

        if not (i % 1000):
            last_loss = running_loss / 1000
            print(f"Batch {i} Loss {last_loss}")
            x_tb = epoch_idx * len(train_loader) + i + 1
            summary_writer.add_scalar("Loss/Train", last_loss, x_tb)

    return last_loss


def per_epoch_activity(train_loader, val_loader, device, optimizer, model, loss_fn, summary_writer,
                       epochs, timestamps):
    """
    each epoch:
    1. perform validataion
    2. save model
    """
    best_val_loss = 1_000_000
    for epoch in range(epochs):

        # 1. per from validation
        model.train(True)

        avg_loss = train_one_epoch(train_loader, device, optimizer, model, loss_fn, summary_writer, epoch)

        model.train(False)
        running_val_loss = 0.0
        for i, data in enumerate(val_loader):
            val_inputs, val_labels = data
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            val_loss = model(val_inputs)
            running_val_loss += val_loss

        avg_val_loss = running_val_loss / (i + 1)
        print(f"Training Loss {avg_loss}, Validation Loss {avg_val_loss}")
        summary_writer.add_scalar("Training/Validation",
                                  {"Training": avg_loss, "Validation": avg_val_loss}, epoch)
        summary_writer.flush()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), fr"saved_model\full_model\model_{epoch}_{timestamps}")


