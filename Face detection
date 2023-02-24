import cv2
import os

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize empty lists for the images and their corresponding labels
images = []
labels = []

# Define a function to load the images and their corresponding labels
def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image using the face detection classifier
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                # Crop the image to just the face region
                face = gray[y:y+h, x:x+w]
                # Resize the face image to a fixed size (e.g. 100x100 pixels)
                face = cv2.resize(face, (100, 100))
                # Add the face image and its corresponding label to the lists
                images.append(face)
                labels.append(label)

# Load the images and their corresponding labels from the two folders (one for each class)
load_images_from_folder('path/to/folder/with/class1/images', 0)
load_images_from_folder('path/to/folder/with/class2/images', 1)

# Convert the lists of images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the model architecture (e.g. a simple Convolutional Neural Network)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
