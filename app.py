import cv2
import mediapipe as mp
import numpy as np
import math

# --------------------
#   GLOBAL SETTINGS
# --------------------

# Adjust the distance threshold (in pixels) for detectiqng a "pinch."
# You may need to experiment with this value depending on your camera resolution/distance.
PINCH_THRESHOLD = 40

# We’ll store box data in a list of dicts. Each dict has:
#   "x": top-left x-coordinate
#   "y": top-left y-coordinate
#   "w": box width
#   "h": box height
#   "color": (B, G, R)
boxes = [
    {"x":  50, "y":  50, "w": 100, "h": 100, "color": (255,   0,   0)},  # Blue
    {"x": 200, "y":  50, "w": 100, "h": 100, "color": (  0, 255,   0)},  # Green
    {"x": 350, "y":  50, "w": 100, "h": 100, "color": (  0,   0, 255)},  # Red
    {"x":  50, "y": 200, "w": 100, "h": 100, "color": (255, 255,   0)},  # Cyan
    {"x": 200, "y": 200, "w": 100, "h": 100, "color": (255,   0, 255)},  # Magenta
    {"x": 350, "y": 200, "w": 100, "h": 100, "color": (  0, 255, 255)},  # Yellow
]

# This variable tracks which box (index) is currently "picked up"
# If -1, no box is selected
selected_box_index = -1

# Offset for smooth dragging (distance between pinch point & box top-left corner)
offset_x = 0
offset_y = 0

# --------------------
#   HELPER FUNCTIONS
# --------------------

def euclidean_distance(x1, y1, x2, y2):
    """Returns the Euclidean distance between two points (x1, y1) and (x2, y2)."""
    return math.dist((x1, y1), (x2, y2))

def point_in_box(px, py, box):
    """
    Checks if the point (px, py) lies within the boundaries
    of the provided 'box' dict.
    """
    return (box["x"] <= px <= box["x"] + box["w"]) and \
           (box["y"] <= py <= box["y"] + box["h"])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --------------------
#   MAIN LOGIC
# --------------------

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        global selected_box_index, offset_x, offset_y

        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get image dimensions
            height, width, _ = frame.shape

            # Process the frame with MediaPipe
            results = hands.process(frame_rgb)

            # Variables to track pinch state and coordinates
            pinch_detected = False
            pinch_x, pinch_y = -1, -1

            # Extract hand landmarks if present
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Landmarks for thumb tip (4) and index tip (8)
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]

                    # Convert from normalized [0,1] to pixel coordinates
                    thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
                    index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)

                    # Calculate the distance between thumb tip and index tip
                    distance = euclidean_distance(thumb_x, thumb_y, index_x, index_y)

                    # If below threshold, consider it a pinch
                    if distance < PINCH_THRESHOLD:
                        pinch_detected = True
                        # We'll take the midpoint between thumb & index as the "pinch" point
                        pinch_x = (thumb_x + index_x) // 2
                        pinch_y = (thumb_y + index_y) // 2

                    # (Optional) Draw hand landmarks on screen for debugging
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                    )

            # --------------------
            #   BOX DRAG LOGIC
            # --------------------

            if pinch_detected:
                # If we haven't already selected a box, check if the pinch is on one
                if selected_box_index == -1:
                    for i, box in enumerate(boxes):
                        if point_in_box(pinch_x, pinch_y, box):
                            selected_box_index = i
                            # Calculate offsets to avoid jumping effect
                            offset_x = box["x"] - pinch_x
                            offset_y = box["y"] - pinch_y
                            break
                else:
                    # If a box is already selected, move it with the pinch
                    boxes[selected_box_index]["x"] = pinch_x + offset_x
                    boxes[selected_box_index]["y"] = pinch_y + offset_y
            else:
                # If pinch is not detected, release the box
                selected_box_index = -1

            # --------------------
            #   DRAW BOXES
            # --------------------
            for i, box in enumerate(boxes):
                # Draw each box
                x1, y1 = box["x"], box["y"]
                x2, y2 = x1 + box["w"], y1 + box["h"]

                # Highlight the selected box (optional)
                thickness = -1  # Filled box
                if i == selected_box_index:
                    # Change color or add outline to show it's selected
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    # Draw an inner rectangle with the box color
                    cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), box["color"], -1)
                else:
                    # Normal (filled) rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box["color"], thickness)

            # Show FPS / Info on screen (optional)
            cv2.putText(
                frame, 
                "Press 'q' to quit", 
                (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )

            # Display final frame
            cv2.imshow("Virtual Pinch-and-Drop", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

from flask import Flask, Response, render_template
# import cv2
# import mediapipe as mp
# import numpy as np
# import math

# app = Flask(__name__)

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# PINCH_THRESHOLD = 40

# boxes = [
#   {"x":  50, "y":  50, "w": 100, "h": 100, "color": (255,   0,   0)},  # Blue
#   {"x": 200, "y":  50, "w": 100, "h": 100, "color": (  0, 255,   0)},  # Green
#   {"x": 350, "y":  50, "w": 100, "h": 100, "color": (  0,   0, 255)},  # Red
#     {"x":  50, "y": 200, "w": 100, "h": 100, "color": (255, 255,   0)},  # Cyan
#    {"x": 200, "y": 200, "w": 100, "h": 100, "color": (255,   0, 255)},  # Magenta
#    {"x": 350, "y": 200, "w": 100, "h": 100, "color": (  0, 255, 255)},  # Yellow
# ]

# selected_box_index = -1
# offset_x = 0
# offset_y = 0

# def euclidean_distance(x1, y1, x2, y2):
#     """Compute Euclidean distance between two points"""
#     return math.dist((x1, y1), (x2, y2))

# def point_in_box(px, py, box):
#     """Check if a point is inside a box"""
#     return (box["x"] <= px <= box["x"] + box["w"]) and \
#            (box["y"] <= py <= box["y"] + box["h"])

# # Capture video
# cap = cv2.VideoCapture(0)

# def generate_frames():
#     global selected_box_index, offset_x, offset_y
    
#     with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             frame = cv2.flip(frame, 1)
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             height, width, _ = frame.shape
#             results = hands.process(frame_rgb)

#             pinch_detected = False
#             pinch_x, pinch_y = -1, -1

#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     thumb_tip = hand_landmarks.landmark[4]
#                     index_tip = hand_landmarks.landmark[8]

#                     thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
#                     index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)

#                     distance = euclidean_distance(thumb_x, thumb_y, index_x, index_y)

#                     if distance < PINCH_THRESHOLD:
#                         pinch_detected = True
#                         pinch_x = (thumb_x + index_x) // 2
#                         pinch_y = (thumb_y + index_y) // 2

#                     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             if pinch_detected:
#                 if selected_box_index == -1:
#                     for i, box in enumerate(boxes):
#                         if point_in_box(pinch_x, pinch_y, box):
#                             selected_box_index = i
#                             offset_x = box["x"] - pinch_x
#                             offset_y = box["y"] - pinch_y
#                             break
#                 else:
#                     boxes[selected_box_index]["x"] = pinch_x + offset_x
#                     boxes[selected_box_index]["y"] = pinch_y + offset_y
#             else:
#                 selected_box_index = -1

#             for box in boxes:
#                 x1, y1 = box["x"], box["y"]
#                 x2, y2 = x1 + box["w"], y1 + box["h"]
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), box["color"], -1)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame_bytes = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True)
