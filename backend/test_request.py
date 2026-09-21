import requests
import cv2
import numpy as np

# Create an in-memory sample image
img = np.ones((200, 500, 3), dtype=np.uint8) * 255
cv2.putText(img, "Hello World Test", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
_, encoded = cv2.imencode(".png", img)

files = {"file": ("test.png", encoded.tobytes(), "image/png")}
data = {"student_id": "TEST101", "writer_name": "Test Student", "class_id": "7A"}

try:
    res = requests.post("http://localhost:8000/api/process-image", data=data, files=files, timeout=40)
    print("STATUS CODE:", res.status_code)
    print("RESPONSE JSON:", res.json())
except Exception as e:
    print("ERROR:", e)
