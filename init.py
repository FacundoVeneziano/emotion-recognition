import cv2
import numpy as np

# Load the face and eye cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Define the emotions that the model can detect
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the region of interest (ROI) containing the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Loop through each eye
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Resize the ROI to match the input size of the model
        roi_color_resized = cv2.resize(roi_color, (300, 300))

        # Convert the image to a 4D tensor (batch size, channels, height, width)
        blob = cv2.dnn.blobFromImage(roi_color_resized, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the model to get the predicted emotion
        net.setInput(blob)
        preds = net.forward()
        label = emotions[preds[0].argmax()]
        pred_probs = preds[0]
        print(pred_probs)

        # Get the index of the predicted emotion with the highest probability
        pred_index = pred_probs.argmax()

        # Get the predicted emotion label
        label = emotions[pred_index]

        # Put the predicted emotion text on the frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
