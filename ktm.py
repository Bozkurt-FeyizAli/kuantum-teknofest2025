
from deepface import DeepFace

import numpy as np
# import pandas as pd
import os

from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.datasets import make_classification
import torch
from torch.utils.data import DataLoader, TensorDataset
from qiskit import QuantumCircuit
# #from qiskit.providers.aer import AerSimulator
# from qiskit_machine_learning.neural_networks import TwoLayerQNN
# from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
# from qiskit.utils import QuantumInstance
from qiskit.circuit.library import RealAmplitudes
# #from qiskit.algorithms.optimizers import COBYLA
# from sklearn.metrics import accuracy_score

# only deep face ile face boundary çıkarma from folder Dataset/train
def extract_faces_from_folder(folder_path):
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
                    # Use DeepFace to detect and extract the face
                    face_img_list = DeepFace.represent(file_path,  enforce_detection=True, model_name="Facenet512", detector_backend='mtcnn')
                    for face_img in face_img_list:
                        print(face_img.get("facial_area"))
                        if face_img is not None:
                            X.append(face_img.get("embedding"))  
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
# face boundary çıkarma ve
# # DeepFace kullanarak yüzleri çıkar
# X_train, y_train = extract_faces_from_folder('Dataset/train')
# X_test, y_test = extract_faces_from_folder('Dataset/test')

# np.save('X_faces_train.npy', X_train)
# np.save('y_labels_train.npy', y_train)
# np.save('X_faces_test.npy', X_test)
# np.save('y_labels_test.npy', y_test)

#write data into a file for future use

# use saved data
X_train_orig = np.load('X_faces_train.npy')
y_train = np.load('Y_labels_train.npy')

X_test_orig = np.load('X_faces_test.npy')
y_test = np.load('y_labels_test.npy')


# --- Özellik Ölçekleme ---
pca = PCA(n_components=N_QUBITS)
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

# DataLoader, verileri batch'ler halinde modele beslememizi sağlar.
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

print(f"Eğitim verisi boyutu: {X_train.shape}")
print(f"Test verisi boyutu: {X_test.shape}")

from qiskit.circuit import ParameterVector

# --- Adım 3: Kuantum Veri Kodlama (Feature Map) ---
feature_params = ParameterVector('x', length=N_QUBITS)

# ŞİMDİ, bu parametreleri kullanacağımız boş devreyi oluşturuyoruz.
feature_map = QuantumCircuit(N_QUBITS, name="FeatureMap")
for i in range(N_QUBITS):
    feature_map.ry(feature_params[i], i)

# feature_map.draw('mpl', style='iqx') # Bu satır hala çalışır.
from qiskit_machine_learning.neural_networks import EstimatorQNN
# --- Adım 4: Parametreli Kuantum Devresi (Ansatz) ---
ansatz = RealAmplitudes(N_QUBITS, reps=2, entanglement='linear')
# ansatz.draw('mpl', style='iqx')


# Kuantum Devresini Oluşturma
qc = QuantumCircuit(N_QUBITS)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)
print("\nTam Kuantum Devresi (Feature Map + Ansatz):")
# qc.draw('mpl', style='iqx')


# --- QNN'i Qiskit'te Tanımlama ---
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp

observable_strings = ['I'*i + 'Z' + 'I'*(N_QUBITS-1-i) for i in range(N_QUBITS)]

observables = [SparsePauliOp(s) for s in observable_strings]

qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    observables=observables,  
    estimator=Estimator() 
)
print(f"\nQNN Giriş Parametreleri: {qnn.input_params}")
print(f"QNN Ağırlık Parametreleri: {qnn.weight_params}")

from qiskit_machine_learning.connectors import TorchConnector
from sklearn.metrics import accuracy_score
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn as nn


class HybridModel(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        # Qiskit QNN'ini bir PyTorch katmanına dönüştürüyoruz.
        self.qnn_layer = TorchConnector(qnn)
        
        # Kuantum katmanın çıktısını (N_QUBITS boyutlu) alıp,
        # N_CLASSES boyutlu bir çıktıya dönüştüren klasik katman.
        self.classical_layer = nn.Linear(N_QUBITS, N_CLASSES)

    def forward(self, x):
        # Veri akışı: Girdi -> Kuantum Katman -> Klasik Katman -> Çıktı
        x = self.qnn_layer(x)
        x = self.classical_layer(x)
        return x

# Modeli oluşturma
model = HybridModel(qnn)
print("\nHibrit Model Mimarisi:")
print(model)


# --- Adım 6: Eğitim Döngüsü ---
loss_fn = nn.CrossEntropyLoss()  # Sınıflandırma için kayıp fonksiyonu
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

history = {'loss': [], 'accuracy': []}

print("\nEğitim Başlatılıyor...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()          # Gradyanları sıfırla
        preds = model(batch_x)         # Tahmin yap
        loss = loss_fn(preds, batch_y) # Kaybı hesapla
        loss.backward()                # Geri yayılım (Parameter-Shift Rule Qiskit tarafından yönetilir)
        optimizer.step()               # Parametreleri güncelle
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    history['loss'].append(avg_loss)
    
    # Her epoch sonunda test doğruluğunu kontrol et
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_x, batch_y in test_loader:
            preds = model(batch_x)
            predicted_labels = torch.argmax(preds, dim=1)
            total += batch_y.size(0)
            correct += (predicted_labels == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        history['accuracy'].append(accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}], Kayıp (Loss): {avg_loss:.4f}, Test Doğruluğu: {accuracy:.2f}%")


# ==============================================================================
# 4. SONUÇLARIN DEĞERLENDİRİLMESİ
# ==============================================================================
print("\nEğitim Tamamlandı!")

# Eğitim sürecini görselleştirme
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(history['loss'])
ax1.set_title("Eğitim Kaybı (Loss)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")

ax2.plot(history['accuracy'])
ax2.set_title("Test Verisi Doğruluğu (Accuracy)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()

# Final test performansı
model.eval()
with torch.no_grad():
    y_pred_list = []
    for batch_x, _ in test_loader:
        preds = model(batch_x)
        y_pred_list.extend(torch.argmax(preds, dim=1).numpy())

final_accuracy = accuracy_score(y_test, y_pred_list)
print(f"\nModelin Nihai Test Doğruluğu: {final_accuracy*100:.2f}%")