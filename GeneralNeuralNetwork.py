import torch
from torch import nn
import qiskit
import qiskit_machine_learning
import qiskit_algorithms
import qiskit_optimizers
import qiskit_terra
import qiskit_aer
import qiskit_ignis
import qiskit_machine_learning
import qiskit_machine_learning_algorithms
import qiskit_machine_learning_algorithms_



class GeneralNeuralNetwork(nn.Module):
    def __init__(self, Qbit_number, hiddenlayer_number, entanglement_type):
        qc = QuantumCircuit(Qbit_number)
        feature_map = ZZFeatureMap(feature_dimension=Qbit_number)
        ansatz = RealAmplitudes(num_qubits=Qbit_number)

        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )

        self.qnn_layer = TorchConnector(qnn)


    def forward(self, x):
        x = self.qnn_layer(x)
        return x

    def train_model(self, learning_rate, EPOCHS, train_loader, test_loader):
        loss_fn = nn.CrossEntropyLoss()  # Sınıflandırma için kayıp fonksiyonu
        optimizer = Adam(self.parameters(), lr=learning_rate)
            
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
