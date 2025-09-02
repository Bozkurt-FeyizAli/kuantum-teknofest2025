
from deepface import DeepFace

import numpy as np
# import pandas as pd
import os

from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.datasets import make_classification
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from qiskit import QuantumCircuit, Aer, transpile
# #from qiskit.providers.aer import AerSimulator
# from qiskit_machine_learning.neural_networks import TwoLayerQNN
# from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
# from qiskit.utils import QuantumInstance
# from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
# #from qiskit.algorithms.optimizers import COBYLA
# from sklearn.metrics import accuracy_score

# only deep face ile face boundary çıkarma from folder Dataset/train
import cv2
def extract_faces_from_folder(folder_path, target_size=(64, 64)):
    X = []
    y = []
    label_map = {}  # Map to convert folder names to numeric labels
    current_label = 0

    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)
        if os.path.isdir(subdir_path):
            if subdir not in label_map:
                label_map[subdir] = current_label
                current_label += 1
            label = label_map[subdir]

            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                try:
                    # Use DeepFace to detect and extract the face
                    face_img = DeepFace.represent(file_path,  enforce_detection=True, model_name="Facenet512", detector_backend='mtcnn')
                    for face in face_img:
                        print(face.get("facial_area"))
                        if face_img is not None:
                            X.append(face.get("embedding"))  # Flatten the image to a 1D array
                            y.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y)

def  MinMaxScaler():
    pass

N_SAMPLES = 1800       # Toplam veri örneği sayısı
N_FEATURES_INITIAL = 512 # Klasik ön işleme (CNN) sonrası özellik sayısı (simülasyon)
N_CLASSES = 8          # Sınıf (oda) sayısı
N_QUBITS = 4           # Kuantum devresinde kullanılacak kübit sayısı

SHOTS = 1024           # Ölçüm tekrar sayısı (gürültüyü azaltmak için)
EPOCHS = 30            # Eğitim döngüsü sayısı
LEARNING_RATE = 0.01   # Öğrenme oranı
BATCH_SIZE = 16        # Eğitimde aynı anda işlenecek veri sayısı
#sahte veri oluşturmak yerine face boundary çıkarma ve
# DeepFace kullanarak yüzleri çıkar
# Load the dataset and extract faces
folder_path = 'Dataset/train'
X, y = extract_faces_from_folder(folder_path)


print(f"Başlangıç Veri Boyutu (X): {X.shape}")
print(f"Etiket Boyutu (y): {y.shape}")

if X.shape[0] > 0:
    print(f"Extracted {X.shape[0]} faces with {X.shape[1]} features each.")
    print(f"Labels shape: {y.shape}")
else:
    print("No faces were extracted. Check the folder path and image content.")

# --- Adım 2: Özellik Ölçekleme ---
# Özellikleri [0, 1] aralığına ölçekliyoruz.
X = X.reshape(X.shape[0], -1)  # Flatten images if necessary
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(f"Ölçeklenmiş Veri Boyutu (X_scaled): {X_scaled.shape}")

