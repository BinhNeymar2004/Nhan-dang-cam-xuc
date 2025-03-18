import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load mô hình đã huấn luyện
model = load_model("emotion_model.keras")


# Nhãn cảm xúc từ mô hình
emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Chọn ảnh đầu vào
image_path = "test13.jpg"  # Thay bằng đường dẫn ảnh
image = cv2.imread(image_path)

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces, gray

def reduce_brightness_spots(image):
    # Giảm độ sáng của các vùng quá sáng
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.medianBlur(l, 5)  # Làm mịn vùng sáng quá mức
    adjusted_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

if image is None:
    print("⚠️ Không thể đọc ảnh!")
else:
    faces, gray = detect_faces(image)
    
    # Nếu không phát hiện khuôn mặt, thử điều chỉnh ảnh và phát hiện lại
    if len(faces) == 0:
        print("⚠️ Không phát hiện khuôn mặt, thử điều chỉnh ảnh...")
        image = reduce_brightness_spots(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces, gray = detect_faces(image)
    
    if len(faces) == 0:
        print("⚠️ Không phát hiện khuôn mặt sau khi điều chỉnh!")
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = img_to_array(face) / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            # Dự đoán cảm xúc
            emotion_prediction = model.predict(face)[0]
            max_index = np.argmax(emotion_prediction)
            emotion_label = emotion_labels.get(max_index, "Không xác định")

            # Vẽ khung và nhãn
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, emotion_label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Hiển thị ảnh và biểu đồ trong cùng một cửa sổ
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Hiển thị ảnh với khung nhận diện
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].axis("off")
        ax[0].set_title("Nhận diện cảm xúc")
        
        # Hiển thị biểu đồ
        sns.barplot(ax=ax[1], x=list(emotion_labels.values()), y=emotion_prediction)
        ax[1].set_xlabel("Cảm xúc")
        ax[1].set_ylabel("Tỉ lệ dự đoán (%)")
        ax[1].set_title("Biểu đồ tỉ lệ dự đoán cảm xúc")
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()