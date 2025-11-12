# debug_loader.py
import sys
sys.path.insert(0, '.')

from torchvision import datasets
from ddpm import script_utils
from torch.utils.data import DataLoader

print("[1] Loading dataset...")
dataset = datasets.CIFAR10(
    root='./cifar_train',
    train=True,
    download=False,
    transform=script_utils.get_transform(),
)
print("[2] Dataset loaded. Creating DataLoader...")

# 关键：显式设 num_workers=0
loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
print("[3] DataLoader created. Fetching first batch...")

try:
    x, y = next(iter(loader))
    print(f"[✅] Success! x.shape = {x.shape}, y.shape = {y.shape}")
except Exception as e:
    print(f"[❌] Error: {e}")
    import traceback
    traceback.print_exc()