# cnn_pretrained_comparison.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import time, os

# ---------------- BUILD SIMPLE CNN ----------------
def build_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3),  # (1 input, 32 filters, 3x3 kernel)
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 13 * 13, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

# ---------------- DATA LOADER ----------------
def load_data():
    tf = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST("data", train=False, transform=tf)
    fashion = datasets.FashionMNIST("data", train=False, transform=tf)
    return mnist, fashion

# ---------------- EVALUATE ----------------
def evaluate(model, loader, device, is_pretrained=False):
    model.eval()
    y_true, y_pred = [], []
    start = time.time()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # For pretrained models (AlexNet, MobileNetV2)
            if is_pretrained:
                # Convert grayscale → RGB if needed
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                # Resize from 28×28 → 224×224
                x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

            out = model(x)
            _, p = torch.max(out, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(p.cpu().numpy())

    t = time.time() - start
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, t, cm



# ---------------- PLOT CONFUSION MATRICES ----------------
def plot_cm(cms, titles, dataset_name, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, len(cms), figsize=(18, 4))
    for ax, cm, title in zip(axs, cms, titles):
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_cnn_comparison.png"), dpi=300)
    plt.close()

# ---------------- FINE-TUNE ----------------
def fine_tune(model, train_loader, device, epochs=2, lr=1e-4):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            if x.shape[2] < 224:
                x = F.interpolate(x, size=(224, 224), mode='bilinear')
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Fine-tune Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

# ---------------- MAIN ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Load MNIST and FashionMNIST datasets
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST("data", train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST("data", train=False, transform=transform)
    fashion_train = datasets.FashionMNIST("data", train=True, transform=transform, download=True)
    fashion_test = datasets.FashionMNIST("data", train=False, transform=transform)

    loaders = {
        "MNIST": (
            torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True),
            torch.utils.data.DataLoader(mnist_test, batch_size=256, shuffle=False)
        ),
        "FashionMNIST": (
            torch.utils.data.DataLoader(fashion_train, batch_size=128, shuffle=True),
            torch.utils.data.DataLoader(fashion_test, batch_size=256, shuffle=False)
        )
    }

    # Base CNN model
    cnn = build_cnn().to(device)
    cnn.load_state_dict(torch.load("models/cnn_model_noclass.pth", map_location=device))

    # Pretrained AlexNet
    alex = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
    alex.classifier[6] = nn.Linear(4096, 10)

    # Pretrained MobileNetV2
    mobile = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
    mobile.classifier[1] = nn.Linear(1280, 10)

    # Iterate over both datasets
    for name, (train_loader, test_loader) in loaders.items():
        print(f"\n--- {name} ---")
        cms, titles = [], []

        # Evaluate base CNN
        acc, f1, t, cm = evaluate(cnn, test_loader, device)
        print(f"CNN: Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s, Params={sum(p.numel() for p in cnn.parameters())/1e6:.2f}M")
        cms.append(cm); titles.append("CNN")

        # Evaluate pretrained AlexNet   
        acc, f1, t, cm = evaluate(alex, test_loader, device, is_pretrained=True)
        print(f"AlexNet (Pretrained): Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s, Params={sum(p.numel() for p in alex.parameters())/1e6:.2f}M")
        cms.append(cm); titles.append("AlexNet Pretrained")

        # Evaluate pretrained MobileNetV2
        acc, f1, t, cm = evaluate(mobile, test_loader, device, is_pretrained=True)
        print(f"MobileNetV2 (Pretrained): Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s, Params={sum(p.numel() for p in mobile.parameters())/1e6:.2f}M")
        cms.append(cm); titles.append("MobileNetV2 Pretrained")

        # Fine-tuned AlexNet
        alex_ft_path = f"models/alexnet_finetuned_{name}.pth"
        alex_ft = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
        alex_ft.classifier[6] = nn.Linear(4096, 10)

        if os.path.exists(alex_ft_path):
            alex_ft.load_state_dict(torch.load(alex_ft_path, map_location=device))
            print("Loaded fine-tuned AlexNet.")
        else:
            print("Fine-tuning AlexNet...")
            fine_tune(alex_ft, train_loader, device, epochs=2)
            torch.save(alex_ft.state_dict(), alex_ft_path)
            print("Fine-tuned AlexNet saved.")

        acc, f1, t, cm = evaluate(alex_ft, test_loader, device, is_pretrained=True)
        print(f"AlexNet (Fine-tuned): Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s")
        cms.append(cm); titles.append("AlexNet Fine-tuned")

        # # Fine-tuned MobileNetV2
        # mobile_ft_path = f"models/mobilenet_finetuned_{name}.pth"
        # mobile_ft = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
        # mobile_ft.classifier[1] = nn.Linear(1280, 10)

        # if os.path.exists(mobile_ft_path):
        #     mobile_ft.load_state_dict(torch.load(mobile_ft_path, map_location=device))
        #     print("Loaded fine-tuned MobileNetV2.")
        # else:
        #     print("Fine-tuning MobileNetV2...")
        #     fine_tune(mobile_ft, train_loader, device, epochs=2)
        #     torch.save(mobile_ft.state_dict(), mobile_ft_path)
        #     print("Fine-tuned MobileNetV2 saved.")

        # acc, f1, t, cm = evaluate(mobile_ft, test_loader, device, is_pretrained=True)
        # print(f"MobileNetV2 (Fine-tuned): Acc={acc:.4f}, F1={f1:.4f}, Time={t:.2f}s")
        # cms.append(cm); titles.append("MobileNetV2 Fine-tuned")

        # Plot and save all confusion matrices for this dataset

        # Skip fine-tuning MobileNetV2 to save time
        print("Skipping MobileNetV2 fine-tuning to save time.")
        
        plot_cm(cms, titles, name, save_dir="outputs")


if __name__ == "__main__":
    main()
