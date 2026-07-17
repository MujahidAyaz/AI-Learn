"""
============================================================
Fashion-MNIST Image Classification using PyTorch
============================================================

Project Entry Point

Responsibilities
----------------
1. Prepare project environment
2. Load dataset
3. Display dataset summary
4. Visualize sample images
============================================================
"""

# ==========================================================
# Import Project Modules
# ==========================================================

from configs.config import DEVICE
from src.dataset import create_dataloaders
from src.utils import (
    create_directories,
    set_random_seed,
    display_dataset_info,
    visualize_samples,
)


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 60)
    print("Fashion-MNIST Image Classification")
    print("=" * 60)

    create_directories()

    set_random_seed()

    print(f"\nUsing Device : {DEVICE}")

    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders()

    display_dataset_info(train_dataset, test_dataset)

    visualize_samples(train_dataset)

    print("\nProject setup completed successfully!")


# ==========================================================
# Program Entry Point
# ==========================================================

if __name__ == "__main__":
    main()