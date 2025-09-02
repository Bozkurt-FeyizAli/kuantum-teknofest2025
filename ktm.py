
from deepface import DeepFace

import numpy as np
# import pandas as pd
import os

from qiskit import QuantumCircuit
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
def extract_faces_from_folder(folder_path, target_size=(64, 64)):
    X = []
    y = []
    label_map = {} 
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
                    # Usage of DeepFace to detect and extract the face
                    face_img = DeepFace.represent(file_path,  enforce_detection=True, model_name="Facenet512", detector_backend='mtcnn')
                    for face in face_img:
                        print(face.get("facial_area"))
                        if face_img is not None:
                            X.append(face.get("embedding"))  
                            y.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y)

N_SAMPLES = 1800      
N_FEATURES_INITIAL = 512 
N_CLASSES = 8        
N_QUBITS = 4          

SHOTS = 1024          
EPOCHS = 30           
LEARNING_RATE = 0.01  
BATCH_SIZE = 16       
#sahte veri oluşturmak yerine face boundary çıkarma ve
# DeepFace kullanarak yüzleri çıkar

folder_path = 'Dataset/train'
X, y = extract_faces_from_folder(folder_path)

#write data into a file for future use
np.save('X_faces.npy', X)
np.save('y_labels.npy', y)

# use saved data
X = np.load('X_faces.npy')
y = np.load('y_labels.npy')


print(f"Başlangıç Veri Boyutu (X): {X.shape}")
print(f"Etiket Boyutu (y): {y.shape}")

if X.shape[0] > 0:
    print(f"Extracted {X.shape[0]} faces with {X.shape[1]} features each.")
    print(f"Labels shape: {y.shape}")
else:
    print("No faces were extracted. Check the folder path and image content.")

# --- Adım 2: Özellik Ölçekleme --- with PCA for 6 feature
# Özellikleri [0, 1] aralığına ölçekliyoruz.
pca = PCA(n_components=N_QUBITS)
X_pca = pca.fit_transform(X)
print(f"PCA sonrası Veri Boyutu (X_pca): {X_pca.shape}")
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_pca)
print(f"Ölçeklendirilmiş Veri Boyutu (X_scaled): {X_scaled.shape}")

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X_pca)

# Veri setini eğitim ve test olarak ikiye ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# PyTorch ile çalışmak için verileri Tensor formatına dönüştürüyoruz.
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# DataLoader, verileri batch'ler halinde modele beslememizi sağlar.
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE)

print(f"Eğitim verisi boyutu: {X_train.shape}")
print(f"Test verisi boyutu: {X_test.shape}")

from qiskit.circuit import ParameterVector

# --- Adım 3: Kuantum Veri Kodlama (Feature Map) ---
# Açı kodlama (Angle Encoding) stratejisini uyguluyoruz.

# ÖNCE, N_QUBITS uzunluğunda bir parametre vektörü oluşturuyoruz.
# Bu bize x[0], x[1], x[2], x[3] gibi parametreler verir.
feature_params = ParameterVector('x', length=N_QUBITS)



# ŞİMDİ, bu parametreleri kullanacağımız boş devreyi oluşturuyoruz.
feature_map = QuantumCircuit(N_QUBITS, name="FeatureMap")
for i in range(N_QUBITS):
    # Önceden oluşturduğumuz parametreleri sırayla Ry kapılarına atıyoruz.
    feature_map.ry(feature_params[i], i)

# feature_map.draw('mpl', style='iqx') # Bu satır hala çalışır.
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.circuit.library import QNNCircuit

from qiskit_machine_learning.neural_networks import EstimatorQNN
# --- Adım 4: Parametreli Kuantum Devresi (Ansatz) ---
# Bu devre, öğrenme işlemini gerçekleştirecek olan eğitilebilir katmandır.
# RealAmplitudes, dönme ve CNOT kapılarından oluşan standart ve etkili bir PQC'dir.
# 'reps=2' ile devreyi sığ tutarak NISQ cihazları için uygun hale getiriyoruz.
ansatz = RealAmplitudes(N_QUBITS, reps=2, entanglement='linear')
ansatz.draw('mpl', style='iqx')


# Kuantum Devresini Oluşturma
qc = QuantumCircuit(N_QUBITS)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)
print("\nTam Kuantum Devresi (Feature Map + Ansatz):")
qc.draw('mpl', style='iqx')


# --- QNN'i Qiskit'te Tanımlama ---
from qiskit.primitives import StatevectorEstimator
# DEĞİŞİKLİK: String'leri Qiskit nesnelerine çevirmek için bu sınıfı import ediyoruz.
from qiskit.quantum_info import SparsePauliOp

# 1. Adım: Önceki gibi string listesini oluşturuyoruz.
observable_strings = ['I'*i + 'Z' + 'I'*(N_QUBITS-1-i) for i in range(N_QUBITS)]

# 2. Adım (YENİ): Şimdi bu string'leri SparsePauliOp nesnelerine dönüştürüyoruz.
observables = [SparsePauliOp(s) for s in observable_strings]

# Artık `observables` listemiz, Qiskit'in backward pass için beklediği doğru türde nesneler içeriyor.
qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    observables=observables,  # Düzeltilmiş, doğru tipteki observable listesini kullanıyoruz
    estimator=StatevectorEstimator() 
)
