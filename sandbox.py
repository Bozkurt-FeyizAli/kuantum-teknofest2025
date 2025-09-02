from typing import Iterable
from deepface import DeepFace
import numpy as np
from sklearn.decomposition import PCA
import os

from sklearn.preprocessing import MinMaxScaler
import torch
import GeneralNeuralNetwork as GNN
from torch.utils.data import DataLoader, TensorDataset
import re


# import cv2
# import matplotlib.pyplot as plt

def extract_faces_from_folder(folder_path, target_size=(64, 64)):
    X = []

    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Use DeepFace to detect and extract the face
            face_img = DeepFace.represent(file_path,  enforce_detection=True, model_name="Facenet512", detector_backend='mtcnn')
            for face in face_img:
                print(face.get("facial_area"))
                if face_img is not None:
                    #writen into file rather than memory
                
                    X.append(face.get("embedding"))  # Flatten the image to a 1D array
                    np.save("face_embedding.npy", face.get("embedding"))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    


    return np.array(X)

# X= extract_faces_from_folder("Dataset/val/Person_7")

# pca_faces = PCA(n_components=8).fit_transform(X)
# print(pca_faces)
# print(pca_faces.shape)


# #minmax scaler
# minmax_scaler = MinMaxScaler()
# pca_faces_scaled = minmax_scaler.fit_transform()
# print(pca_faces_scaled)
# print(pca_faces_scaled.shape)

# face_objs = DeepFace.extract_faces(
#   img_path = "two-face.jpg", detector_backend = "opencv", align = True
# )



# for face_obj in face_objs:
#     print("------------------------------------------------------------------------------------------------")
#     print(DeepFace.represent(img_path=face_obj["face"], model_name="Facenet512", enforce_detection=False))
#     print("------------------------------------------------------------------------------------------------")

X_train_orig = np.load('X_faces_train.npy')
y_train = np.load('Y_labels_train.npy')

X_test_orig = np.load('X_faces_test.npy')
y_test = np.load('y_labels_test.npy')


# --- Özellik Ölçekleme ---
pca = PCA(n_components=6)
X_train_pca = pca.fit_transform(X_train_orig)
X_test_pca = pca.transform(X_test_orig)
print(f"PCA sonrası Veri Boyutu (X_train_pca): {X_train_pca.shape}")


scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train_scaled = scaler.fit_transform(X_train_pca)
X_test_scaled = scaler.transform(X_test_pca)
print(f"Ölçeklendirilmiş Veri Boyutu (X_train_scaled): {X_train_scaled.shape}")


# PyTorch ile çalışmak için verileri Tensor formatına dönüştürüyoruz.
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)


model = GNN.GeneralNeuralNetwork(Qbit_number=6,
hiddenlayer_number=2,
entanglement_type="linear")

# parameters=[[2,3]*8]  # 8 qubit, 2 hidden layer
# seed = torch.tensor(parameters, dtype=torch.float32)
#get parameters from file model_parameters.txt
#Traceback (most recent call last):
#   File "/home/feyiz-ali/Masaüstü/3. sınıf/projeler/kuantum-teknofest2025/sandbox.py", line 101, in <module>
#     parameters = [list(map(float, line.strip().split())) for line in f.readlines()]
#                   ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ValueError: could not convert string to float: 'tensor(['
with open("model_parameters.txt", "r") as f:
    content = f.read()

# Use regex to find all numbers (including negative and scientific notation)
numbers_str = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", content)
numbers = [float(num) for num in numbers_str]
parameters = [numbers]  # Wrap in another list to create a 2D structure

seed = torch.tensor(parameters, dtype=torch.float32)
model.train_model(learning_rate=0.01, EPOCHS=10, train_loader=train_loader , test_loader=test_loader, seed=seed)