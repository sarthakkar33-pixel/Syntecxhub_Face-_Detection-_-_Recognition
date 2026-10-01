import cv2
import os

name = input("Enter person's name: ")

folder = os.path.join("known_faces", name)
os.makedirs(folder, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0

print("Look at the camera")
print("Press SPACE to capture")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Register Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):

        if len(faces) > 0:

            x, y, w, h = faces[0]

            face = gray[y:y+h, x:x+w]

            count += 1

            filename = os.path.join(
                folder,
                f"{count}.jpg"
            )

            cv2.imwrite(filename, face)

            print("Saved:", filename)

        else:
            print("No face detected!")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nRegistration completed!")
print("Total images:", count)