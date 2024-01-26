import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Inicjalizacja detektora twarzy za pomocą kaskady Haara
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ścieżka do foldera z danymi
data_dir = "E:\RAFAL\Pwr\MAGISTER\\2SEMESTR\GSN\data"
class_names = os.listdir(data_dir)

images = []
labels = []
img_size = (128, 128)  # Docelowy rozmiar zdjęć

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        # Zmniejszenie obrazu do docelowych rozmiarów (128x128)
        resized_img = cv2.resize(img, img_size)

        # Wyświetlenie obrazu z naniesioną ramką twarzy (800x800)
        enlarged_img = cv2.resize(img, (800, 800))
        # gray = cv2.cvtColor(enlarged_img, cv2.COLOR_BGR2GRAY) <- Usuwamy konwersję na skalę szarości
        faces = face_cascade.detectMultiScale(enlarged_img, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))
        for (x, y, w, h) in faces:
            cv2.rectangle(enlarged_img, (x, y), (x+w, y+h), (255, 0, 0), 10)
            # Wyizolowanie obszaru twarzy
            face_img = enlarged_img[y:y+h, x:x+w]  # Zapisanie obszaru twarzy z oryginalnego kolorowego obrazu
            face_img_resized = cv2.resize(face_img, img_size)  # Dopasowanie rozmiaru wyciętej twarzy
            # # Wyświetlenie obrazu twarzy z ramką          
            cv2.imshow('Detected Faces', enlarged_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # Wyświetlenie wyciętego obszaru twarzy obok obrazu z ramką
            cv2.imshow('Cropped Face', face_img_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            images.append(face_img_resized)
            labels.append(class_names.index(class_name))

# Zamiana list na tablice numpy
images = np.array(images)
labels = np.array(labels)

# Podział danych na zestawy treningowe i testowe
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Przekształcenie etykiet do formatu używanego przez sieć neuronową
num_classes = len(class_names)
train_labels = to_categorical(train_labels, num_classes=num_classes)
val_labels = to_categorical(val_labels, num_classes=num_classes)

from collections import Counter
# Liczenie ilości wystąpień klas w danych treningowych i walidacyjnych
train_class_counts = Counter(np.argmax(train_labels, axis=1))
val_class_counts = Counter(np.argmax(val_labels, axis=1))

print("Train Class Counts:", train_class_counts)
print("Validation Class Counts:", val_class_counts)


# Definicja modelu sieci neuronowej (Convolutional Neural Network)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

from tensorflow.keras.utils import plot_model
# Generowanie diagramu
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Augmentacja danych
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    zoom_range=0.1
)

datagen.fit(train_images)

# Trenowanie modelu
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) / 32, epochs=50,
                    validation_data=(val_images, val_labels))

import matplotlib.pyplot as plt

# Rysowanie wykresów skuteczności uczenia
plt.figure(figsize=(12, 4))

# Skuteczność uczenia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Funkcja straty
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save("FR.h5")