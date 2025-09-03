import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from qiskit_machine_learning.connectors import TorchConnector

import qiskit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator  # Yerel (sim) için
from qiskit.quantum_info import SparsePauliOp

# GERÇEK DONANIM İÇİN:
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as RuntimeEstimator, Session, Options


class GeneralNeuralNetwork(nn.Module):
    def __init__(
        self,
        Qbit_number,
        hiddenlayer_number,
        entanglement_type,
        seed=None,
        use_ibm=False,
        ibm_backend="ibm_oslo",
        shots=4000,
        optimization_level=1,
        resilience_level=0,
    ):
        super().__init__()
        self.seed = seed
        self.use_ibm = use_ibm
        self.session = None  # IBM Runtime session (varsa) kapanış için tutulur

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Kuantum devresi
        qc = qiskit.QuantumCircuit(Qbit_number)
        feature_params = ParameterVector('x', length=Qbit_number)

        feature_map = qiskit.QuantumCircuit(Qbit_number, name="FeatureMap")
        for i in range(Qbit_number):
            feature_map.ry(feature_params[i], i)

        ansatz = RealAmplitudes(num_qubits=Qbit_number, reps=hiddenlayer_number, entanglement=entanglement_type)

        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        print(qc)
        # qc.draw('mpl')  # İsteğe bağlı

        # Gözlemler: Z_i
        observable_strings = ['I'*i + 'Z' + 'I'*(Qbit_number-1-i) for i in range(Qbit_number)]
        observables = [SparsePauliOp(s) for s in observable_strings]

        # Estimator seçimi: gerçek cihaz (Runtime) ya da yerel
        if self.use_ibm:
            try:
                service = QiskitRuntimeService()  # save_account ile daha önce kaydedilmiş olmalı
            except Exception as e:
                raise RuntimeError(
                    "IBM Quantum hesabına bağlanılamadı. "
                    "Önce bir kere çalıştırın: "
                    'from qiskit_ibm_runtime import QiskitRuntimeService; '
                    'QiskitRuntimeService.save_account(channel="ibm_quantum", token="...")'
                ) from e

            backend = service.backend(ibm_backend)
            self.session = Session(service=service, backend=backend)
            options = Options(
                optimization_level=optimization_level,
                resilience_level=resilience_level,
                execution={"shots": shots},
            )
            estimator_impl = RuntimeEstimator(session=self.session, options=options)
        else:
            estimator_impl = Estimator()  # yerel (simülatör) referans primitive

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            observables=observables,
            estimator=estimator_impl
        )

        self.qnn_layer = TorchConnector(qnn)
        ##self.classical_layer = nn.Linear(Qbit_number, 8)

    def forward(self, x):
        x = self.qnn_layer(x)
        #x = self.classical_layer(x)
        return x

    def train_model(self, learning_rate, EPOCHS, train_loader, test_loader, seed=None):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)

        history = {'loss': [], 'accuracy': []}

        print("\nEğitim Başlatılıyor...")
        for epoch in range(EPOCHS):
            self.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self(batch_x)
                #print pred values
                print("preds:", preds)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history['loss'].append(avg_loss)

            self.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for batch_x, batch_y in test_loader:
                    preds = self(batch_x)
                    
                    print("preds:", preds)
                    predicted_labels = torch.argmax(preds, dim=1)
                    print("preds:", predicted_labels)
                    total += batch_y.size(0)
                    correct += (predicted_labels == batch_y).sum().item()
                    print("batch_y:", batch_y)

                accuracy = 100 * correct / total
                history['accuracy'].append(accuracy)

            with open("model_parameters.txt", "w") as f:
                for param in self.parameters():
                    f.write(f"{param.data}\n")

            with open("training_log.txt", "a") as f:
                f.write(f"Epoch: {epoch+1}, Loss: {avg_loss}, Accuracy: {accuracy}\n")

            print(f"Epoch [{epoch+1}/{EPOCHS}], Kayıp (Loss): {avg_loss:.4f}, Test Doğruluğu: {accuracy:.2f}%")

        # Eğitim bittiğinde session'ı kapatın
        self.close()

    def close(self):
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None
    def evaluate(self,pca_deger,label)->bool:
        self.eval()
        with torch.no_grad():
            preds = self(pca_deger)
            print(preds)
            predicted_label = torch.argmax(preds, dim=0)
            print(f"Predicted: {predicted_label}, Actual: {label}")
            return predicted_label == label
    
    def loadModel(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def validate(self, X_val, y_val):
        PCM = np.zeros(shape=[8,4], dtype=int)  # Assuming 10 classes (0-9)
        ROC = np.zeros(shape=[8,len(X_val),2], dtype=float)  # Assuming 10 classes (0-9)
        self.eval()
        with torch.no_grad():
            record=0
            for record in range(len(X_val)):
                prediction_probility = self(X_val[record])
                predicted_label = torch.argmax(prediction_probility, dim=0)
                actual_label = y_val[record].item()
                if predicted_label == actual_label:
                    PCM[actual_label][0] += 1
                else:
                    PCM[actual_label][2] += 1
                    PCM[predicted_label][1] += 1
                for i in range(len(PCM)):
                    if(i != actual_label and i != predicted_label):
                        PCM[i][3] += 1
                print("Confusion Matrix:")
                print(PCM)
                for i in range(len(PCM)):
                    TPR = PCM[i][0] / (PCM[i][0] + PCM[i][2]) if (PCM[i][0] + PCM[i][2]) > 0 else 0
                    FPR = PCM[i][1] / (PCM[i][1] + PCM[i][3]) if (PCM[i][1] + PCM[i][3]) > 0 else 0
                    ROC[i][record][0] = TPR
                    ROC[i][record][1] = FPR
                record+=1
        print("Final Confusion Matrix:")
        print(PCM)
        print("ROC Data Points:")
        print(ROC)

        #loging into file ROC data
        with open("ROC_data.txt", "w") as f:
            for i in range(len(ROC)):
                f.write(f"Class {i}:\n")
                for j in range(len(ROC[i])):
                    f.write(f"{ROC[i][j][0]}, {ROC[i][j][1]}\n")
                f.write("\n")
        #loging into file PCM data