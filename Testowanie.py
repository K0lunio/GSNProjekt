import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

class_names = ["Damian", "Kuba", "Rafal"]  # Lista nazw klas

# Wczytanie wytrenowanego modelu klasyfikacji twarzy
model = load_model("FR.h5")

# Inicjalizacja detektora twarzy za pomocą kaskady Haara
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Rozpocznij strumień z kamery (numer 0 oznacza domyślną kamerę)
cap = cv2.VideoCapture(0)

# Inicjalizacja słownika przechowującego etykiety i predykcje dla każdej klasy
class_samples = {class_name: {'true_labels': [], 'predicted_labels': []} for class_name in class_names}

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))
    
    face_rois = []  
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (128, 128))
        face_rois.append(resized_face)
    
    if face_rois:
        face_rois = np.array(face_rois)  
        face_rois = face_rois / 255.0
        
        predictions = model.predict(face_rois)
        
        for i, prediction in enumerate(predictions):
            predicted_class = np.argmax(prediction)
            class_name = class_names[predicted_class]
          
            x, y, w, h = faces[i]            
          
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, class_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            key = cv2.waitKey(1)  # Przeniesione poza pętlę klasyfikacji
            if key & 0xFF == ord('q'):
                break

            if key == ord('3') and predicted_class == 2:  # Klawisz 3 i etykieta 'Rafal'
                true_label = class_names[predicted_class]
                class_samples['Rafal']['true_labels'].append(true_label)
                class_samples['Rafal']['predicted_labels'].append(class_name)  # Skopiowanie wartości predicted_label
            else:
                class_samples[class_name]['predicted_labels'].append(class_name)  # Skopiowanie wartości predicted_label

    cv2.imshow('Face Detection', frame)

# Po wyjściu z pętli, oblicz metryki dla każdej klasy
for class_name, samples in class_samples.items():
    true_labels = samples['true_labels']
    predicted_labels = samples['predicted_labels']
    
    if len(true_labels) > 0:  # Sprawdzenie, czy lista etykiet jest niepusta
        accuracy = accuracy_score(true_labels, predicted_labels)
        class_report = classification_report(true_labels, predicted_labels, target_names=[class_name])
        print(f"Metryki dla {class_name}:")
        print("Dokładność:", accuracy)
        print("Raport klasyfikacji:")
        print(class_report)
    else:
        print(f"Brak próbek dla klasy {class_name}")

cap.release()
cv2.destroyAllWindows()
