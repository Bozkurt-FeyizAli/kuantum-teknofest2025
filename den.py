# Gerekli Kütüphanelerin Yüklenmesi
import numpy as np
import matplotlib.pyplot as plt

# 1. Klasik Makine Öğrenmesi ve Veri İşleme
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# 2. PyTorch Kütüphaneleri (Klasik ve Hibrit Model için)
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# 3. Qiskit Kütüphaneleri
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

# ÖNEMLİ: Qiskit'in simülatörünü kullanacağız. 
# Gerçek donanımda çalıştırmak için 'qiskit_ibm_provider' gerekir.
from qiskit.primitives import Estimator

print("Tüm kütüphaneler başarıyla yüklendi.")

# ==============================================================================
# 1. PROBLEM PARAMETRELERİ VE VERİ SETİ HAZIRLIĞI
# ==============================================================================
# Yarışma problemine uygun parametreleri burada tanımlıyoruz.
# Bu değerleri değiştirerek deneyler yapabilirsiniz.

N_SAMPLES = 200        # Toplam veri örneği sayısı
N_FEATURES_INITIAL = 64 # Klasik ön işleme (CNN) sonrası özellik sayısı (simülasyon)
N_CLASSES = 8          # Sınıf (oda) sayısı
N_QUBITS = 4           # Kuantum devresinde kullanılacak kübit sayısı. 
                         # PCA sonrası özellik sayısı buna eşit olmalı.

SHOTS = 1024           # Ölçüm tekrar sayısı (gürültüyü azaltmak için)
EPOCHS = 30            # Eğitim döngüsü sayısı
LEARNING_RATE = 0.01   # Öğrenme oranı
BATCH_SIZE = 16        # Eğitimde aynı anda işlenecek veri sayısı

# --- Adım 1: Sahte Veri Seti Üretimi ---
# Bu kısım, yarışmada size verilecek veri setini taklit eder.
# Normalde burada CNN'den çıkan 512 özellikli bir vektör olurdu.
X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES_INITIAL,
    n_informative=8,  # Anlamlı özellik sayısı
    n_redundant=8,
    n_classes=N_CLASSES,
    n_clusters_per_class=1,
    random_state=42
)
print(f"Başlangıç Veri Boyutu (X): {X.shape}")
print(f"Etiket Boyutu (y): {y.shape}")


# --- Adım 2: Klasik Ön İşleme ve Boyut Azaltma (PCA) ---
# Kuantum bilgisayarın işleyebilmesi için özellik sayısını kübit sayısına düşürüyoruz.
pca = PCA(n_components=N_QUBITS)
X_pca = pca.fit_transform(X)
print(f"PCA sonrası Veri Boyutu (X_pca): {X_pca.shape}")

# Verileri kuantum devresine kodlamadan önce [0, pi] aralığına ölçeklendiriyoruz.
# Bu, açı kodlama için standart bir adımdır.
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


# ==============================================================================
# 2. KUANTUM DEVRESİ (QNN) TASARIMI
# ==============================================================================
# Gerekli import'u dosyanın başına eklediğinden emin ol
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
# EstimatorQNN, bir kuantum devresini sinir ağı olarak kullanmamızı sağlar.
# Devrenin beklenen değerlerini hesaplayarak çıktı üretir.
# Çıktı olarak her kübitin Z eksenindeki beklenen değerini ölçeceğiz.
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


# ==============================================================================
# 3. HİBRİT MODELİN TANIMLANMASI VE EĞİTİMİ
# ==============================================================================

# --- Adım 5: Hibrit Model (PyTorch + Qiskit) ---
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