import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the JSON file
JSON_PATH = "hasil.json"

with open(JSON_PATH, "r") as fp:
    data = json.load(fp)

# Extract dataset features and labels
X = np.array(data["MFCCs"])
y = np.array(data["labels"])

# Convert labels to one-hot encoding
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a CNN model
model = models.Sequential()
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=X_train.shape[1:]))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(y_train.shape[1], activation='softmax'))

# Compile the model with RMSprop optimizer
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = callbacks.ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)

# Train the model with callbacks
start_time = time.time()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, callbacks=[early_stopping, model_checkpoint])
end_time = time.time()
training_time = end_time - start_time

# Evaluate the best model
best_model = models.load_model("best_model.h5")
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
predictions = best_model.predict(X_test)
predicted_labels = label_binarizer.inverse_transform(predictions)

# Calculate precision, recall, and f1-score
classification_rep = classification_report(label_binarizer.inverse_transform(y_test), predicted_labels, target_names=data["mapping"], output_dict=True)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(label_binarizer.inverse_transform(y_test), predicted_labels)

# Save the classification results to a JSON file
classification_results = {
    "accuracy": test_accuracy,
    "epoch": len(history.history['accuracy']),
    "training_time": training_time,
    "precision": classification_rep,
}
with open("hasil_klasifikasi_rmsprop.json", "w") as fp:
    json.dump(classification_results, fp, indent=4)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print(f"Accuracy: {test_accuracy}")
print(f"Training Time: {training_time} seconds")
print(classification_rep)