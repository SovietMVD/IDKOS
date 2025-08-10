import os
import cv2
import torch
from PIL import Image
import model
from torchvision import models
import Isolation_Kernel as IK
import torchvision.transforms as transforms
import numpy as np
from sklearn.decomposition import PCA
import Check as Ch
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((700, 700)),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # 归一化到 [-1, 1]
])
transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((500,500)),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # 归一化到 [-1, 1]
])

Device='cuda'
def image_loader(root_dir):
    Images_Tensor = []
    Data=[]
    image_data_list=[]
    image_data_tensor=[]
    for image_name in os.listdir(root_dir):
            image_name_dir = os.path.join(root_dir, image_name)
            for image in os.listdir(image_name_dir):
                image_dir = os.path.join(image_name_dir, image)
                with Image.open(image_dir) as img:
                    img = img.convert('RGB')
                if image==image_name:
                    image_tensor=transform(img)
                else:
                    image_tensor=transform_1(img)
                x=torch.unsqueeze(image_tensor,0)
                image_data_tensor.append([image,x,image_name])
                image_data_list.append([image,img,image_name])
            Images_Tensor.append(image_data_tensor)
            Data.append(image_data_list)
            image_data_tensor=[]
            image_data_list=[]
    return Images_Tensor,Data
def Abnormal_Identity_NSet(Images_Tensor, ROOT_DATA,save_dir):
    imgs = []
    Abnormal_Image = []
    one_image=[]
    output_name='error_dir.txt'
    output_file_dir=os.path.join(save_dir,output_name)
    with torch.no_grad():
        for list in Images_Tensor:
            e=0
            for sublist in list:
                img = sublist[1]
                img = img.to(Device)
                if sublist[0]==sublist[2]:
                    _,output = model(img)
                else:
                    _,output = model_1(img)
                one_image.append(output.cpu())
                e+=1
                if e==6:
                    break
            one_image_tensor=torch.cat(one_image,dim=0)
            one_image_tensor = one_image_tensor.view(1, -1)
            one_image=[]
            imgs.append(one_image_tensor)
        imgs_tensor = torch.cat(imgs, dim=0)
        imgs_numpy = imgs_tensor.numpy()
        pca = PCA(n_components=None)
        imgs_pca = pca.fit_transform(imgs_numpy)
        imgs_pca_tensor = torch.tensor(imgs_pca)
        row_avgs, index_list = IK.IKSimilarity(imgs_pca_tensor, Sdata=None, psi=psi, t=t, )
        for index in range(len(index_list)):
            Abnormal_Image.append([ROOT_DATA[index_list[index]][0][1],ROOT_DATA[index_list[index]][0][2]])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(output_file_dir):
            open(output_file_dir, 'w').close()
        exist_name=set()
        with open(output_file_dir, 'r') as f:
            exist_name=set(line.strip() for line in f.readlines())
        for i in range(len(Abnormal_Image)):
            with open(output_file_dir, 'a') as f:
                if Abnormal_Image[i][1] not in exist_name:
                    f.write(Abnormal_Image[i][1]+'\n')


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
    New_image_root='IDKOS/bind_train_test_slice'
    save_error = "ERROR22"
    New_image_dir="dataset/NEW_IMAGE"
    Images_Tensor = []
    New_image=[]
    Root_Dir = 'dataset/text feature/bind_train_test_slice'
    model = models.resnet50(pretrained=True)
    model_1=models.resnet50(pretrained=True)
    num_classes = len(os.listdir(Root_Dir))
    model = model.to(Device)
    model_1=model_1.to(Device)
    model.load_state_dict(torch.load("MODEL/model_train_identity_full_image(resnet).pth", weights_only=True))
    model_1.load_state_dict(torch.load("MODEL/model_train_identity_slice.pth", weights_only=True))
    model.eval()
    Epoch = 0
    i=0
    image_three=[]
    x_three=[]
    x_full_tensor=[]
    image_full_tensor=[]
    for image_name in os.listdir(New_image_root):
        image_name_dir = os.path.join(New_image_root, image_name)
        for image in os.listdir(image_name_dir):
            image_dir = os.path.join(image_name_dir, image)
            with Image.open(image_dir) as img:
                img = img.convert('RGB')
            if image == image_name:
                image_tensor = transform(img)
            else:
                image_tensor = transform_1(img)
            x = torch.unsqueeze(image_tensor, 0)
            x_full_tensor.append([image,x,image_name])
            image_full_tensor.append([image, img, image_name])
        x_three.append(x_full_tensor)
        image_three.append(image_full_tensor)
        x_full_tensor=[]
        image_full_tensor=[]
        i+=1
        if i%3==0 or i==len(os.listdir(New_image_root)):
            for Person in os.listdir(Root_Dir):
                Person_dir = os.path.join(Root_Dir, Person)
                save_error_dir = os.path.join(save_error, Person)
                Images_Tensor,Data=image_loader(Person_dir)
                Images_Tensor.extend(x_three)
                Data.extend(image_three)
                psi = int(len(Images_Tensor) / 3)
                print("psi:", psi)
                t = 200
                print("t:", t)
                Epoch += 1
                print("Epoch:", Epoch)
                Abnormal_Identity_NSet(Images_Tensor,Data,save_error_dir)
            image_three=[]
            x_three=[]
    num_classes = len(os.listdir(save_error))
    new_class_folders = classify(save_error, num_classes)
    labels, scores, auroc, aupr, fpr95 = compare_folders(New_image_root, new_class_folders)

    print(f"AUROC: {auroc:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"FPR95: {fpr95:.4f}")


    plot_metrics(labels, scores, auroc, aupr)
