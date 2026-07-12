validation_losses = [
    0.90,
    0.70,
    0.55,
    0.42,
    0.36,
    0.34,
    0.35,
    0.37,
    0.40,
    0.43,
]

best_loss = float("inf")
patience = 3
counter = 0

for epoch, loss in enumerate(validation_losses, start=1):

    print(f"Epoch {epoch:2d} | Validation Loss: {loss:.2f}")

    if loss < best_loss:

        best_loss = loss
        counter = 0

        print(" New best model found.")

    else:

        counter += 1

        print(f" No improvement ({counter}/{patience})")

        if counter >= patience:

            print("\nEarly stopping triggered.")
            break