import copy
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# Hardware & resource constraints (Strict VPS anti-OOM rules)
torch.set_num_threads(2)
DEVICE = torch.device("cpu")
BATCH_SIZE = 8
EPOCHS = 100
PATIENCE = 10
LR = 1e-4
NUM_WORKERS = 0
PIN_MEMORY = False
DPI = 300
MIN_RAM_MB = 200
MIN_DISK_GB = 3.0

PROJECT_DIR = Path("/root/rice_mobilenetv3")
DATA_DIR = PROJECT_DIR / "dataset"
RUNS_DIR = PROJECT_DIR / "runs/mobilenetv3_cpu"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = RUNS_DIR / "checkpoint_latest.pth"
BEST_MODEL_PATH = RUNS_DIR / "best_mobilenetv3.pth"
PROGRESS_FILE = RUNS_DIR / "training_progress.json"

process = psutil.Process(os.getpid())
peak_rss_mb = 0.0


def log_msg(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)


def check_watchdog(batch_idx):
    global peak_rss_mb
    rss = process.memory_info().rss / (1024 * 1024)
    if rss > peak_rss_mb:
        peak_rss_mb = rss

    vm = psutil.virtual_memory()
    available_mb = vm.available / (1024 * 1024)
    free_disk_gb = shutil.disk_usage(PROJECT_DIR).free / (1024 ** 3)

    if available_mb < MIN_RAM_MB:
        msg = f"[WATCHDOG TRIGGERED] Sisa RAM rendah: {available_mb:.1f} MB (< {MIN_RAM_MB} MB). RSS saat ini: {rss:.1f} MB."
        log_msg(msg)
        return False, msg

    if free_disk_gb < MIN_DISK_GB:
        msg = f"[WATCHDOG TRIGGERED] Sisa disk rendah: {free_disk_gb:.2f} GB (< {MIN_DISK_GB} GB)."
        log_msg(msg)
        return False, msg

    return True, None


def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    total_imgs = len(dataset)
    if total_imgs == 0:
        raise ValueError(f"Dataset kosong di: {DATA_DIR}")

    train_size = int(0.8 * total_imgs)
    val_size = total_imgs - train_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    log_msg(f"Dataset dimuat: {total_imgs} gambar | Train: {train_size} | Val: {val_size} | Kelas: {dataset.classes}")
    return train_loader, val_loader, dataset.classes


def build_model(num_classes):
    log_msg("Membangun model MobileNetV3-Small (weights='DEFAULT')...")
    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    if total == 0:
        return 0.0, np.array([], dtype=int), np.array([], dtype=int)

    return correct / total, np.array(y_true), np.array(y_pred)


def save_final_metrics(model, val_loader, class_names, total_training_time):
    log_msg("Menghitung metrik evaluasi akhir...")
    accuracy, y_true, y_pred = evaluate(model, val_loader)

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    start_inf = time.time()
    evaluate(model, val_loader)
    inference_time = (time.time() - start_inf) / max(len(y_true), 1)

    torch.save(model.state_dict(), BEST_MODEL_PATH)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(RUNS_DIR / "confusion_matrix.csv")

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(RUNS_DIR / "classification_report.csv")

    summary = pd.DataFrame(
        {
            "Accuracy": [accuracy],
            "Precision_macro": [precision],
            "Recall_macro": [recall],
            "F1_macro": [f1],
            "Inference_Time_per_image_sec": [inference_time],
            "Training_Time_sec": [total_training_time],
            "Best_Val_Accuracy": [accuracy],
            "Peak_RSS_MB": [peak_rss_mb],
        }
    )
    summary.to_csv(RUNS_DIR / "summary_metrics.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - MobileNetV3 Small")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(class_names)), class_names)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(RUNS_DIR / "confusion_matrix_300dpi.png", dpi=DPI)
    plt.close()

    log_msg(f"Evaluasi selesai. Disimpan di: {RUNS_DIR}")
    print(summary, flush=True)


def main():
    global peak_rss_mb
    log_msg("========== MEMULAI TRAINING MOBILENETV3-SMALL ==========")
    train_loader, val_loader, class_names = get_dataloaders()
    model = build_model(len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    best_acc = 0.0
    patience_counter = 0
    start_epoch = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    accumulated_time = 0.0

    if CHECKPOINT_PATH.exists():
        log_msg(f"Ditemukan checkpoint di {CHECKPOINT_PATH}, memuat status sebelumnya...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_acc = ckpt.get("best_acc", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        accumulated_time = ckpt.get("training_time", 0.0)
        if "best_model_weights" in ckpt:
            best_model_weights = ckpt["best_model_weights"]
        log_msg(f"Resume sukses! Mulai dari epoch {start_epoch + 1}/{EPOCHS} (Best Val Acc sebelumnya: {best_acc:.4f})")

    session_start_time = time.time()
    stopped_by_watchdog = False

    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        log_msg(f"\n--- Epoch [{epoch + 1}/{EPOCHS}] ---")

        for b_idx, (images, labels) in enumerate(train_loader):
            if b_idx % 20 == 0 or b_idx == len(train_loader) - 1:
                ok, reason = check_watchdog(b_idx)
                if not ok:
                    log_msg("Darurat: Menghentikan proses loop secara aman...")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_acc": best_acc,
                        "patience_counter": patience_counter,
                        "best_model_weights": best_model_weights,
                        "training_time": accumulated_time + (time.time() - session_start_time),
                        "stopped_reason": reason,
                    }, CHECKPOINT_PATH)
                    gc.collect()
                    stopped_by_watchdog = True
                    break

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item()

            if (b_idx + 1) % 100 == 0 or (b_idx + 1) == len(train_loader):
                cur_rss = process.memory_info().rss / (1024 * 1024)
                vm = psutil.virtual_memory()
                log_msg(f"Epoch {epoch+1} [{b_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f} | RSS: {cur_rss:.1f} MB | Free RAM: {vm.available / (1024*1024):.1f} MB")

        if stopped_by_watchdog:
            log_msg("Training dihentikan oleh watchdog. Checkpoint tersimpan aman.")
            break

        avg_loss = running_loss / len(train_loader)
        val_acc, _, _ = evaluate(model, val_loader)
        scheduler.step(val_acc)

        cur_rss = process.memory_info().rss / (1024 * 1024)
        if cur_rss > peak_rss_mb:
            peak_rss_mb = cur_rss

        epoch_time = time.time() - epoch_start_time
        log_msg(
            f"Epoch [{epoch + 1}] Selesai ({epoch_time:.1f}s) | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Best: {best_acc:.4f} | RSS: {cur_rss:.1f} MB"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            log_msg(f"Model terbaik baru disimpan (Acc: {best_acc:.4f})")
            torch.save(best_model_weights, BEST_MODEL_PATH)
        else:
            patience_counter += 1
            log_msg(f"Tidak ada peningkatan ({patience_counter}/{PATIENCE})")

        total_elapsed = accumulated_time + (time.time() - session_start_time)
        # Simpan checkpoint epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "patience_counter": patience_counter,
            "best_model_weights": best_model_weights,
            "training_time": total_elapsed,
        }, CHECKPOINT_PATH)

        # Update json progress
        vm = psutil.virtual_memory()
        with open(PROGRESS_FILE, "w") as fp:
            json.dump({
                "epoch": epoch + 1,
                "epochs_total": EPOCHS,
                "train_loss": round(avg_loss, 4),
                "val_acc": round(val_acc, 4),
                "best_acc": round(best_acc, 4),
                "patience": f"{patience_counter}/{PATIENCE}",
                "peak_rss_mb": round(peak_rss_mb, 1),
                "available_ram_mb": round(vm.available / (1024 * 1024), 1),
                "swap_used_mb": round(vm.used / (1024 * 1024), 1),
                "total_time_sec": round(total_elapsed, 1),
            }, fp, indent=2)

        gc.collect()

        if patience_counter >= PATIENCE:
            log_msg("\nEarly stopping teraktivasi.")
            break

    total_training_time = accumulated_time + (time.time() - session_start_time)
    log_msg("\n========== TRAINING SELESAI ==========")
    log_msg(f"Total Waktu Training: {total_training_time:.2f} detik")
    log_msg(f"Best Validation Accuracy: {best_acc:.4f}")

    if not stopped_by_watchdog:
        model.load_state_dict(best_model_weights)
        save_final_metrics(model, val_loader, class_names, total_training_time)


if __name__ == "__main__":
    main()
