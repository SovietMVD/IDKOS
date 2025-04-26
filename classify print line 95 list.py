import os
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import numpy as np
import re
import matplotlib.pyplot as plt


def classify(Error_dir, num_class):
    new_class_folders = []
    i = 0

    for person in os.listdir(Error_dir):
        person_dir = os.path.join(Error_dir, person)
        file_path = os.path.join(person_dir, 'error_dir.txt')

        with open(file_path, 'r') as file:
            for line in file:
                folder_name = line.strip()
                if folder_name:
                    found = False
                    for item in new_class_folders:
                        if item[0] == folder_name:
                            item[1] += 1
                            found = True
                            break
                    if not found:
                        new_class_folders.append([folder_name, 1])
        i += 1

    # Calculate proportion
    total = num_class
    for item in new_class_folders:
        count = item[1]
        proportion = round(count / total, 4)
        item[1] = proportion

    return new_class_folders


def compare_folders(folder_root, new_class_folders):#统计函数
    labels = []
    scores = []

    folder_proportions = {item[0]: item[1] for item in new_class_folders}

    for folder_name in os.listdir(folder_root):
        folder_path = os.path.join(folder_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label = 1 if 'New' in folder_name else 0
        labels.append(label)
        original_name = re.sub(r'_New$', '', folder_name)
        score = folder_proportions.get(original_name, 0.0)
        scores.append(score)

    labels = np.array(labels)
    scores = np.array(scores)
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    desired_tpr = 0.95
    if np.max(tpr) >= desired_tpr:
        fpr95 = np.interp(desired_tpr, tpr, fpr)
    else:
        fpr95 = 1.0

    return labels, scores, auroc, aupr, fpr95


def plot_metrics(labels, scores, auroc, aupr):
    plt.figure(figsize=(12, 6))

    # ROC curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # Precision-Recall curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AP = {aupr:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    New_image_root = 'IDKOS/New_class(both_image)'
    save_error = "dataset/ERROR_dir"

    num_classes = len(os.listdir(save_error))
    new_class_folders = classify(save_error, num_classes)
    labels, scores, auroc, aupr, fpr95 = compare_folders(New_image_root, new_class_folders)

    print(f"AUROC: {auroc:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"FPR95: {fpr95:.4f}")

    plot_metrics(labels, scores, auroc, aupr)