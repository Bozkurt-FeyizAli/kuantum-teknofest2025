from typing import Iterable
from deepface import DeepFace
import numpy as np
from sklearn.decomposition import PCA
import os

from sklearn.preprocessing import MinMaxScaler
import torch
import GeneralNeuralNetwork as GNN
from torch.utils.data import DataLoader, TensorDataset
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

X_train = np.load('X_faces_train.npy')
y_train = np.load('Y_labels_train.npy')

X_test = np.load('X_faces_test.npy')
y_test = np.load('y_labels_test.npy')

# PyTorch ile çalışmak için verileri Tensor formatına dönüştürüyoruz.
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=16)

model = GNN.GeneralNeuralNetwork(Qbit_number=8, hiddenlayer_number=2, entanglement_type="linear")

# parameters=[[2,3]*8]  # 8 qubit, 2 hidden layer
# seed = torch.tensor(parameters, dtype=torch.float32)
model.train_model(learning_rate=0.01, EPOCHS=10, train_loader=None, test_loader=None)