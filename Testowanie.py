import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
import os

class_names = ["Damian", "Kuba", "Rafal"]  # Lista nazw klas

# Wczytanie wytrenowanego modelu klasyfikacji twarzy
model = load_model('FR.h5')

# Inicjalizacja detektora twarzy za pomocą kaskady Haara
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Rozpocznij strumień z kamery (numer 0 oznacza domyślną kamerę)
cap = cv2.VideoCapture(0)

# Inicjalizacja list przechowujących etykiety i predykcje
true_labels = []  # Etykiety przypisane przez Ciebie
predicted_labels = []  # Predykcje modelu

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

        if key == ord('1'):
            true_label = "Damian"
        elif key == ord('2'):
            true_label = "Kuba"
        elif key == ord('3'):
            true_label = "Rafal"  
        else:
            true_label = None  # Ustawienie true_label na None, gdy nie przypisano żadnej klasy

        if true_label is not None:
            true_labels.append(true_label)
            predicted_labels.append(class_name)
    
    cv2.imshow('Face Detection', frame)

# Po wyjściu z pętli, oblicz metryki
accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
print(f"Dokładność modelu: {accuracy}")

from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(true_labels, predicted_labels, labels=class_names)
print("Macierz pomyłek:")
print(confusion_mat)

from sklearn.metrics import classification_report

class_report = classification_report(true_labels, predicted_labels, target_names=class_names)
print("Raport klasyfikacji:")
print(class_report)

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(true_labels, predicted_labels, average=None, labels=class_names)
recall = recall_score(true_labels, predicted_labels, average=None, labels=class_names)
f1 = f1_score(true_labels, predicted_labels, average=None, labels=class_names)

for i, class_name in enumerate(class_names):
    print(f"Klasa: {class_name}")
    print(f"  Precyzja: {precision[i]}")
    print(f"  Czułość: {recall[i]}")
    print(f"  F1-score: {f1[i]}")

cap.release()
cv2.destroyAllWindows()
