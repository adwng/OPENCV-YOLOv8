from ultralytics import YOLO
import cv2
import random

# Initialize Global Variables
# Predict on image
model = YOLO(
    r"..\runs/detect/train/weights/best.pt"
)

# Paths for Video
videoPath = [
    '16 legs (Imperfect).avi',
    '16 legs (Perfect).avi',
    '14 legs.avi',
    '8 legs.avi'
]


def run():
    # Choose Function Based on User's Input
    print("\n1. Training")
    print("2. Predict")
    print("3. Leave")
    choice = int(input("Please Select: (1/2/3):"))
    if choice == 1:
        training()
    elif choice == 2:
        # Select Video Based on User's Input, will return value
        print("\n1. 16 Legs (Imperfect) [0]")
        print("2. 16 Legs (Perfect) [1]")
        print("3. 14 Legs (Perfect) [2]")
        print("4. 8 Legs (Perfect) [3]")
        video = int(input("Please Select Your Video of Choice: "))
        if video not in [0, 1, 2, 3]:
            run()
        else:
            predict(video)
    else:
        return


def training():
    # Training downloaded Roboflow Dataset
    path = r'C:\Users\andre\OneDrive - Asia Pacific University\Degree Year Three (2)\Machine Visioon & Intellegience\Assignment Code\ic-body-leg-identification.v2i.yolov8\data.yaml'
    trainingModel = YOLO('yolov8n.yaml')  # creates a new model

    results = trainingModel.train(data=path, epochs=100)
    return


def predict(selection):
    # Predict selected videos and count legs and show overall confidence score

    # Video capture from file
    cap = cv2.VideoCapture(videoPath[selection])

    while cap.isOpened():
        legList = []
        legConfList = []

        ret, frame = cap.read()

        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 480))

        results = model.predict(resized_frame, line_width=1, scale=0.2)

        annotated_frame = results[0].plot()  # Get annotated frame

        result = results[0]  # Used to extract detected class IDs' and confidence

        # Extract information coordinates, class ID, and confidence from results
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            conf = round(box.conf[0].item(), 2)

            if class_id == "ic-legs":
                # Append total number of class ID "ic-legs" and their confidence to array
                legList.append(class_id)
                legConfList.append(conf)

        # FInd total number of detected Legs using len() and averaged the total confidence by the legs
        text = f"IC-legs: {len(legList)}: {round(sum(legConfList) / len(legConfList), 2)}"

        # Get the text size and baseline
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size[0], text_size[1]

        # Calculate the position for bottom-right corner
        bottom_right_x = resized_frame.shape[1] - 40
        bottom_right_y = resized_frame.shape[0] - 40

        # Put the text in the bottom-right corner
        cv2.putText(
            annotated_frame,
            text,
            (bottom_right_x - text_width, bottom_right_y),
            font, font_scale,
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            thickness
        )

        cv2.imshow("Predicted Image", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    run()
