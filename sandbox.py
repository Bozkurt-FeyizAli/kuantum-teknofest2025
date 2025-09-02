from deepface import DeepFace
import numpy as np
from sklearn.decomposition import PCA
import os
# import cv2
# import matplotlib.pyplot as plt

def normalize_faces(X):

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

X= extract_faces_from_folder("Dataset/val/Person_7")

pca_faces = PCA(n_components=8).fit_transform(X)
print(pca_faces)
print(pca_faces.shape)


#minmax scaler
minmax_scaler = MinMaxScaler()
pca_faces_scaled = minmax_scaler.fit_transform()
print(pca_faces_scaled)
print(pca_faces_scaled.shape)



# face_objs = DeepFace.extract_faces(
#   img_path = "two-face.jpg", detector_backend = "opencv", align = True
# )



# for face_obj in face_objs:
#     print("------------------------------------------------------------------------------------------------")
#     print(DeepFace.represent(img_path=face_obj["face"], model_name="Facenet512", enforce_detection=False))
#     print("------------------------------------------------------------------------------------------------")
