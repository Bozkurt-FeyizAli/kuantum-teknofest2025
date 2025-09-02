import os
from deepface import DeepFace
import numpy as np

# extract and save face boundary as image from folder 
def extract_faces_from_folder(folder_path):
    face_boundaries = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Use DeepFace to detect and extract the face
            face_img_list = DeepFace.represent(file_path,  enforce_detection=True, model_name="Facenet512", detector_backend='mtcnn')
            for face_img in face_img_list:
                print(face_img.get("facial_area"))
                if face_img is not None:
                    face_boundaries.append(face_img.get("facial_area"))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return np.array(face_boundaries)

X= extract_faces_from_folder("Dataset/val/Person_7")
print(X)
print(X.shape)
np.save("face_boundaries.npy", X)