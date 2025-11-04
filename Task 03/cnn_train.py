# cnn_train_noclass.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time

# ---------------- CNN BUILDER ----------------
def build_cnn():
    cnn = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(32 * 13 * 13, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return cnn

# ---------------- FEATURE EXTRACTOR ----------------
def get_features(model, imgs):
    x = F.relu(model[0](imgs))       # conv
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 32 * 13 * 13)
    x = F.relu(model[4](x))          # 128-neuron layer
    return x

# ---------------- CONFUSION MATRIX ----------------
def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------------- t-SNE ----------------
def tsne_plot(features, labels, title, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating t-SNE for {title} ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=10)
    plt.legend(*sc.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, title.replace(" ", "_") + ".png"), dpi=300)
    plt.close()

# ---------------- TRAIN ----------------
def train_cnn(model, train_loader, device, epochs=5, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f}")

# ---------------- EVALUATE ----------------
def evaluate(model, loader, device, dataset_name, save_dir):
    model.eval()
    preds, true_labels, feats = [], [], []
    start_time = time.time()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            pred = outputs.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            feats.extend(get_features(model, imgs).cpu().numpy())

    end_time = time.time()
    accuracy = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    inference_time = end_time - start_time

    cm_path = os.path.join(save_dir, f"CNN_{dataset_name}_confusion.png")
    plot_confusion_matrix(true_labels, preds, f"CNN on {dataset_name}", cm_path)

    print(f"{dataset_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {inference_time:.2f}s")
    return np.array(feats), np.array(true_labels), accuracy, f1, inference_time

# ---------------- MAIN ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root="data", train=False, transform=transform)
    fashion_test = datasets.FashionMNIST(root="data", train=False, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
    mnist_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False)
    fashion_loader = DataLoader(fashion_test, batch_size=1000, shuffle=False)

    model_path = "models/cnn_model_noclass.pth"
    cnn = build_cnn().to(device)

    if os.path.exists(model_path):
        cnn.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained CNN model.\n")
    else:
        print("Training CNN model on MNIST...")
        train_cnn(cnn, train_loader, device)
        torch.save(cnn.state_dict(), model_path)
        print("Model saved.\n")

    save_dir = "outputs"

    mnist_feats, mnist_lbl, mnist_acc, mnist_f1, mnist_time = evaluate(cnn, mnist_loader, device, "MNIST", save_dir)
    fashion_feats, fashion_lbl, fashion_acc, fashion_f1, fashion_time = evaluate(cnn, fashion_loader, device, "FashionMNIST", save_dir)

    # ---- t-SNE plots ----
    tsne_plot(mnist_feats[:2000], mnist_lbl[:2000], "CNN MNIST 128-layer tSNE", save_dir)
    tsne_plot(fashion_feats[:2000], fashion_lbl[:2000], "CNN FashionMNIST 128-layer tSNE", save_dir)

    # ---- Save results ----
    with open(os.path.join(save_dir, "CNN_results.txt"), "w") as f:
        f.write("=== CNN Results (No-Class Version) ===\n")
        f.write(f"MNIST  -> Accuracy: {mnist_acc:.4f}, F1: {mnist_f1:.4f}, Time: {mnist_time:.2f}s\n")
        f.write(f"Fashion-> Accuracy: {fashion_acc:.4f}, F1: {fashion_f1:.4f}, Time: {fashion_time:.2f}s\n")

if __name__ == "__main__":
    main()
