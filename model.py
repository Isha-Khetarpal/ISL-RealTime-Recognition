import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# path for datasets

train_dataset_path = r'D:\SIH\dataset\Indian'
test_dataset_path = r'D:\SIH\dataset\Indian'
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# training


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(35, activation='softmax'))  # Adjusted to match the number of classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=2,  # Reduced epochs
    validation_data=test_generator
)

# Save the trained model
model.save('ISL_Classification_Model.h5')




















# # Plot training and validation accuracy/loss
# plt.figure(figsize=(18, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training Accuracy vs Validation Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss vs Validation Loss')
# plt.legend()

# # Save the plots
# plt.savefig('ISL_Training_Graph.svg')
# plt.show()

# # # Evaluate the model using a confusion matrix
# # test_data, true_labels = next(test_generator)
# # predictions = model.predict(test_data)
# # predicted_labels = np.argmax(predictions, axis=1)
# # conf_matrix = confusion_matrix(np.argmax(true_labels, axis=1), predicted_labels)

# # plt.figure(figsize=(8, 8))
# # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(35))
# # disp.plot(cmap='Blues', values_format='d')
# # plt.title('Confusion Matrix for ISL Detection')
# # plt.savefig('ISL_Confusion_Matrix.svg')
# # plt.show()


