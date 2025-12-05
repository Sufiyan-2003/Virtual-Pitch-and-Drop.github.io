✋🎨 Virtual Pinch-and-Drag Interaction System

A computer-vision based interactive system that allows users to grab, drag, and move virtual objects on the screen using only hand gestures.
Using MediaPipe Hands + OpenCV, this project detects a pinch gesture (thumb + index finger) and lets the user pick up colored boxes and move them around—just like a real touch interface, but completely touchless.

🚀 Features
✅ 1. Real-Time Pinch Gesture Detection

Detects the distance between thumb tip (4) and index fingertip (8)

When the distance is below the threshold → Pinch = Grab object

✅ 2. Drag Virtual Objects with Hand Movement

Six colored boxes appear on the screen

User can grab a box by pinching over it

Move your hand → the box follows smoothly

Release pinch → box drops in place

✅ 3. Multiple UI Objects

Pre-added draggable color blocks:

Blue

Green

Red

Cyan

Magenta

Yellow

Each block has independent position tracking.

✅ 4. Smooth Dragging with Offset Logic

To prevent jumpy movement:

The system calculates an offset between pinch point and box top-left corner

Ensures natural and stable drag behavior

✅ 5. MediaPipe Hand Tracking

Real-time, high-accuracy hand landmark detection

21 key hand points

Optional drawing of landmarks and hand skeleton

✅ 6. Flask Web Version (Optional)

The project also includes a Flask streaming version:

Runs the virtual drag interaction in browser

Streams camera via /video_feed

Good for remote access and web UI integrations

🧠 How It Works
🔹 Pinch Detection Algorithm
distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)

if distance < PINCH_THRESHOLD:
    pinch_detected = True

🔹 Selecting a Box

When a pinch happens:

Check if pinch point lies inside any box

If yes → mark that box as selected

Store the offset so the box doesn’t jump

🔹 Dragging a Box
box["x"] = pinch_x + offset_x
box["y"] = pinch_y + offset_y

🔹 Dropping a Box

Release pinch → selection resets

Box stays in the last position

🛠️ Technologies Used
Technology	Purpose
OpenCV	Webcam input, rendering, UI drawing
MediaPipe Hands	Hand & gesture tracking
NumPy	Coordinate operations
Math Module	Distance calculation
Flask (Optional)	Browser-based video streaming
📦 Project Structure
.
├── main.py                 # Core pinch-to-drag interaction logic
├── flask_app.py            # Flask streaming version (optional)
├── templates/
│   └── index.html          # Web interface for Flask app
└── README.md

▶️ Running the Project
1. Install Dependencies
pip install opencv-python mediapipe numpy flask

2. Run the Core Application
python main.py

3. (Optional) Run the Flask Server
python flask_app.py


Then open in browser:

http://127.0.0.1:5000/

🎮 Controls
Gesture	Action
🤏 Pinch (thumb + index close)	Select / Grab box
✋ Move hand while pinched	Drag box
👋 Open hand (no pinch)	Drop box
⌨️ Press q	Quit program
🖼️ Output Window

Shows webcam feed

Displays draggable boxes

Shows hand landmarks (optional)

Real-time gesture interaction
