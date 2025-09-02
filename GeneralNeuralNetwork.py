import torch
import torch.nn as nn
from torch.optim import Adam
from qiskit_machine_learning.connectors import TorchConnector

import qiskit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN





class GeneralNeuralNetwork(nn.Module):
    qnn_layer = None
    classical_layer = None
    def __init__(self, Qbit_number, hiddenlayer_number, entanglement_type):
        super().__init__()
        qc = qiskit.QuantumCircuit(Qbit_number)

        feature_params = ParameterVector('x', length=Qbit_number)

        # ŞİMDİ, bu parametreleri kullanacağımız boş devreyi oluşturuyoruz.
        feature_map = qiskit.QuantumCircuit(Qbit_number, name="FeatureMap")
        for i in range(Qbit_number):
        # Önceden oluşturduğumuz parametreleri sırayla Ry kapılarına atıyoruz.
            feature_map.ry(feature_params[i], i)


        ansatz = RealAmplitudes(num_qubits=Qbit_number, reps=hiddenlayer_number, entanglement=entanglement_type)

        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )

        qnn_layer = TorchConnector(qnn)
        classical_layer = nn.Linear(Qbit_number, 2)  # 2 sınıf için çıktı katmanı

    def forward(self, x):
        x = self.qnn_layer(x)
        x = self.classical_layer(x)
        return x

    def train_model(self, learning_rate, EPOCHS, train_loader, test_loader):
        loss_fn = nn.CrossEntropyLoss()  # Sınıflandırma için kayıp fonksiyonu
        optimizer = Adam(self.parameters(),lr=learning_rate)
            
        history = {'loss': [], 'accuracy': []}

        print("\nEğitim Başlatılıyor...")
        for epoch in range(EPOCHS):
            self.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()          # Gradyanları sıfırla
                preds = self(batch_x)         # Tahmin yap
                loss = loss_fn(preds, batch_y) # Kaybı hesapla
                loss.backward()                # Geri yayılım (Parameter-Shift Rule Qiskit tarafından yönetilir)
                optimizer.step()               # Parametreleri güncelle
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            history['loss'].append(avg_loss)
            
            # Her epoch sonunda test doğruluğunu kontrol et
            self.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for batch_x, batch_y in test_loader:
                    preds = self(batch_x)
                    predicted_labels = torch.argmax(preds, dim=1)
                    total += batch_y.size(0)
                    correct += (predicted_labels == batch_y).sum().item()
                
                accuracy = 100 * correct / total
                history['accuracy'].append(accuracy)

            print(f"Epoch [{epoch+1}/{EPOCHS}], Kayıp (Loss): {avg_loss:.4f}, Test Doğruluğu: {accuracy:.2f}%")
