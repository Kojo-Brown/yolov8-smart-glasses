import speech_recognition as sr
from ultralytics import YOLO
import cv2
import pyttsx3
import time

# Initialize the recognizer for speech recognition
recognizer = sr.Recognizer()

# Initialize the TTS engine
engine = pyttsx3.init()

# Load YOLO models
models = [YOLO("best.pt"), YOLO("yolov8n.pt"),YOLO("custom_yolo.pt") ]
model_names = ["Coin", "General", "Custom"]

# Open webcam
cap = cv2.VideoCapture(0)

# Track what was last spoken to avoid repeating
last_said = ""

# Function to listen for the trigger word
def listen_for_trigger():
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        print("Heard command:", command)
        return "start" in command
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return False
    except sr.RequestError:
        print("Could not request results; check your network connection.")
        return False

# Main loop
while True:
    if listen_for_trigger():
        print("Starting YOLO object detection...")
        engine.say("Starting YOLO object detection")
        engine.runAndWait()

        frame_count = 0
        start_time = time.time()
        fps = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            all_labels = set()
            plotted_frame = frame.copy()

            # Run each model on the same frame
            for model, label in zip(models, model_names):
                results = model.predict(source=frame, save=False, conf=0.5, verbose=False)
                boxes = results[0].boxes

                if boxes is not None and boxes.cls.numel() > 0:
                    classes = boxes.cls.tolist()
                    names = [model.names[int(cls)] for cls in classes]
                    all_labels.update(names)

                    # Overlay current results on frame
                    plotted_frame = results[0].plot(img=plotted_frame)

            # Speak only if new labels detected
            spoken = ", ".join(sorted(all_labels))
            if spoken and spoken != last_said:
                print("Detected:", spoken)
                engine.say(f"I see {spoken}")
                engine.runAndWait()
                last_said = spoken

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

            # Show FPS
            cv2.putText(plotted_frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the result
            cv2.imshow("YOLO Multi-Model Detection", plotted_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# speech recognition
# pip install speechrecognition pyaudio
# audio output
# pip install pyttsx3 