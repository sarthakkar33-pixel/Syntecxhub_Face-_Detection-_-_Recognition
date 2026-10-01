# Face Detection & Recognition

A real-time face detection and recognition system built using **Python and OpenCV**.  
The project detects faces through a webcam, allows users to register faces, and recognizes registered people using the **LBPH (Local Binary Patterns Histograms)** face recognition algorithm.

---

## 📌 Project Overview

This project provides a simple real-time face detection and recognition system.

The system can:

- Detect human faces using OpenCV Haar Cascade.
- Register new people using a webcam.
- Store registered face images.
- Train an LBPH face recognizer.
- Recognize registered people in real time.
- Display the person's name above the detected face.
- Display `Unknown` for unregistered people.
- Detect multiple faces in the camera frame.

The project is designed to be beginner-friendly and can be used as an academic or learning project for Computer Science and Computer Vision.

---

## ✨ Features

### 1. Face Detection
Detects faces from the webcam using the OpenCV Haar Cascade classifier.

### 2. Face Registration
Allows a user to register a new person by capturing multiple face images.

### 3. Face Recognition
Uses the LBPH face recognition algorithm to identify registered people.

### 4. Unknown Face Detection
If a detected face does not match the registered faces, the system displays:

```text
Unknown