import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Qiskit kütüphanelerini import et
# DİKKAT: Aer ve QuantumInstance importları güncellendi/değiştirildi
from qiskit_algorithms.utils import algorithm_globals

from qiskit.primitives import Sampler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA


# --- 1. Veri Yükleme ve Hazırlama ---
def load_images_from_folder(folder):
    """
    Belirtilen klasör yapısından görüntüleri ve etiketleri yükler.
    """
    images = []
    labels = []
    label_map = {}
    current_label = 0
    for person_name in sorted(os.listdir(folder)):
        person_folder = os.path.join(folder, person_name)
        if os.path.isdir(person_folder):
            if person_name not in label_map:
                label_map[person_name] = current_label
                current_label += 1
            for filename in os.listdir(person_folder):
                try:
                    img_path = os.path.join(person_folder, filename)
                    # Görüntüyü siyah-beyaz (grayscale) olarak aç ve yeniden boyutlandır
                    with Image.open(img_path).convert('L') as img:
                        # Daha yüksek çözünürlük daha iyi özellik çıkarımı sağlar
                        img = img.resize((64, 64))  # 50x50'den 64x64'e yükseltildi
                        # Histogram eşitleme ile kontrast iyileştirme
                        img_array = np.array(img)
                        # Basit histogram eşitleme
                        img_eq = np.clip((img_array - img_array.min()) * 255.0 / (img_array.max() - img_array.min()), 0, 255)
                        images.append(img_eq.flatten())
                        labels.append(label_map[person_name])
                except Exception as e:
                    print(f"Hata: {img_path} dosyası okunamadı. {e}")
    
    return np.array(images), np.array(labels), label_map

print("[INFO] Veri setleri yükleniyor...")
# Eğitim ve test verilerini belirtilen klasörlerden yükle
X_train_raw, y_train, label_map = load_images_from_folder('out_data/train')
X_test_raw, y_test, _ = load_images_from_folder('out_data/test')

print(f"[DONE] Veri yüklendi. Sınıflar: {label_map}")
print(f"Eğitim verisi boyutu: {X_train_raw.shape}")
print(f"Test verisi boyutu: {X_test_raw.shape}")

# --- 2. Veri Ön İşleme ve Boyut İndirgeme ---
# Kuantum devresinde kullanılacak özellik sayısı (kübit sayısını belirler)
n_features = 12  # Daha fazla özellik daha iyi sonuç verebilir
print(f"\n[INFO] PCA ile özellik sayısı {n_features}'e indirgeniyor...")

# Önce standardizasyon uygula (daha iyi PCA sonuçları için)
standard_scaler = StandardScaler()
X_train_scaled = standard_scaler.fit_transform(X_train_raw)
X_test_scaled = standard_scaler.transform(X_test_raw)

# PCA modelini standardize edilmiş verilere uygula
# explained_variance_ratio_ ile en bilgilendirici bileşenleri seç
pca = PCA(n_components=min(n_features, X_train_scaled.shape[1]), whiten=True)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"[INFO] PCA açıklanan varyans oranı: {pca.explained_variance_ratio_.sum():.3f}")

# Veriyi 0-π aralığına ölçekle (kuantum rotasyon açıları için daha uygun)
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train = scaler.fit_transform(X_train_pca)
X_test = scaler.transform(X_test_pca)
print("[DONE] PCA ve ölçekleme tamamlandı.")


# --- 3. Kuantum Sınıflandırıcı (VQC) Kurulumu ---
print("\n[INFO] Kuantum Sınıflandırıcı (VQC) hazırlanıyor...")

# Algoritma tekrar edilebilirliği için seed ayarı
seed = 1376
algorithm_globals.random_seed = seed

# Kuantum devresini çalıştıracak simülatörü ayarla (Sampler Primitive)
# GÜNCELLEME: QuantumInstance'in yerini almıştır.
sampler = Sampler()

# Adım 3a: Özellik Haritası (Feature Map)
# Klasik veriyi kuantum durumlarına kodlar.
# n_features kadar kübit kullanır.
# Daha karmaşık dolanıklık deseni ve daha fazla tekrar
feature_map = ZZFeatureMap(
    feature_dimension=n_features, 
    reps=3,  # Daha fazla tekrar
    entanglement='full',  # Tam dolanıklık (linear yerine)
    parameter_prefix='x'
)

# Adım 3b: Varyasyonel Devre (Ansatz)
# Öğrenilebilir parametreleri olan kuantum devresi.
# Daha derin ansatz daha iyi öğrenme kapasitesi sağlar
ansatz = RealAmplitudes(
    num_qubits=n_features, 
    reps=4,  # Daha fazla katman
    entanglement='full',  # Tam dolanıklık
    parameter_prefix='theta'
)

# Adım 3c: Optimize Edici (Optimizer)
# Modelin parametrelerini güncelleyecek algoritma.
# Daha fazla iterasyon ve daha iyi parametreler
optimizer = COBYLA(maxiter=200, disp=True)

# VQC modelini oluştur
# GÜNCELLEME: 'quantum_instance' parametresi 'sampler' ile değiştirildi.
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    sampler=sampler, 
    loss='cross_entropy', # Çok sınıflı problemler için standart kayıp fonksiyonu
)
print("[DONE] VQC modeli oluşturuldu.")


# --- 4. Eğitim ve Test ---
print("\n[INFO] Modelin eğitimi başlıyor... (Bu işlem birkaç dakika sürebilir)")
# Modeli eğitim verisiyle eğit
vqc.fit(X_train, y_train)
print("[DONE] Eğitim tamamlandı.")

# Modelin test verisi üzerindeki doğruluğunu hesapla
score = vqc.score(X_test, y_test)
print(f"\n[RESULT] Kuantum Sınıflandırıcının Test Doğruluğu: {score:.2f}")

# --- 5. Sonuçları Görselleştirme (Opsiyonel) ---
print("\n[INFO] Test setinden bazı tahminler yapılıyor...")

# Test setinden birkaç örnek üzerinde tahmin yap
predictions = vqc.predict(X_test[:10])

# Etiketleri isimlere geri dönüştürmek için ters map oluştur
reverse_label_map = {v: k for k, v in label_map.items()}

# Tahminleri ve gerçek etiketleri göster
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test_raw[i].reshape(64, 64), cmap='gray')  # 50x50'den 64x64'e güncellendi
    pred_name = reverse_label_map[predictions[i]]
    true_name = reverse_label_map[y_test[i]]
    ax.set_title(f"Tahmin: {pred_name}\nGerçek: {true_name}", 
                 color='green' if pred_name == true_name else 'red')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Detaylı performans analizi
# Tüm test seti üzerinde tahmin yap
all_predictions = vqc.predict(X_test)

# Sınıflandırma raporu
print("\n[DETAILED RESULTS] Sınıflandırma Raporu:")
target_names = [reverse_label_map[i] for i in range(len(label_map))]
print(classification_report(y_test, all_predictions, target_names=target_names))

# Confusion Matrix
print("\n[INFO] Karışıklık matrisi oluşturuluyor...")
cm = confusion_matrix(y_test, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Kuantum Sınıflandırıcı - Karışıklık Matrisi')
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.show()
