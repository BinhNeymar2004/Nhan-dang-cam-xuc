import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load mô hình đã huấn luyện
model = load_model("emotion_model.keras")

# Nhãn cảm xúc từ mô hình
emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Mở camera
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("⚠️ Không thể truy cập camera!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = img_to_array(face) / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # Dự đoán cảm xúc
        emotion_prediction = model.predict(face)
        max_index = np.argmax(emotion_prediction[0])
        confidence_score = np.max(emotion_prediction[0]) * 100
        emotion_label = emotion_labels.get(max_index, "Không xác định")

        # Vẽ khung và hiển thị nhãn
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion_label}: {confidence_score:.2f}%", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()