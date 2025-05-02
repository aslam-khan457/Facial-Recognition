import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model once
model = load_model("model/emotiondetector.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Mediapipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_emotions(frame):
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    detected_emotion = None

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            w_box = int(bboxC.width * w)
            h_box = int(bboxC.height * h)

            # Ensure ROI is within frame bounds
            x = max(x, 0)
            y = max(y, 0)
            x2 = min(x + w_box, frame.shape[1])
            y2 = min(y + h_box, frame.shape[0])

            face_roi = frame[y:y2, x:x2]
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            try:
                roi_resized = cv2.resize(gray_roi, (48, 48))
            except:
                continue  # skip if face too small or invalid

            roi_resized = roi_resized.astype("float") / 255.0
            roi_resized = img_to_array(roi_resized)
            roi_resized = np.expand_dims(roi_resized, axis=0)

            preds = model.predict(roi_resized, verbose=0)[0]
            label = emotion_labels[preds.argmax()]
            detected_emotion = label

            # Draw bounding box & label
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame, detected_emotion
