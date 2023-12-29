import cv2
import numpy as np


def count():

    videoPath = [
        '16 legs (Imperfect).avi',
        '16 legs (Perfect).avi',
        '14 legs.avi'
    ]

    print("\n1. 16 Legs (Imperfect) [0]")
    print("2. 16 Legs (Perfect) [1]")
    print("3. 14 Legs (Perfect) [2]")
    choice = int(input("Please Select Your Video of Choice:"))

    # Video capture from file
    cap = cv2.VideoCapture(videoPath[choice])  # Replace with your video file test1, test2, test5

    # Declaring global variables
    max_missing_legs = 0
    max_body_area = 0
    max_expected_legs = 0
    ic_body_size = 0
    ratio = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to grayscale
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image_gray, (11, 11), 0)

        # Thresholding
        retval, thresh = cv2.threshold(blurred, 160, 200, cv2.THRESH_BINARY)
        cv2.imshow("Threshold", thresh)

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        cv2.imshow("Erosion", dilation)

        # Find contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        filtered_contours_leg = [cnt for cnt in contours if 50 < cv2.contourArea(cnt) < 1400]
        filtered_contours_body = [cnt2 for cnt2 in contours if 1400 < cv2.contourArea(cnt2) < 150000]

        # Draw bounding boxes around the filtered contours (legs)
        image = frame.copy()

        for contour in filtered_contours_leg:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the number of legs based on the number of filtered contours detected
        number_of_legs = len(filtered_contours_leg)

        for contourB in filtered_contours_body:
            ic_body_contour = max(filtered_contours_body, key=cv2.contourArea)
            cv2.drawContours(image, [ic_body_contour], -1, (0, 255, 0), 2)
            x_body, y_body, w_body, h_body = cv2.boundingRect(ic_body_contour)

            # Calculate the size of the IC body based on the bounding box
            # Assuming the size estimation as the area of the bounding box
            ic_body_size = w_body * h_body

        if choice in [0, 1]:
            ratio = 5700
        elif choice == 2:
            ratio = 6600

        # Track the maximum body area and corresponding expected legs
        if ic_body_size > max_body_area:
            max_body_area = ic_body_size
            standard_leg_to_body_ratio = ratio * (1000000 // ic_body_size)  # Adjust this ratio
            max_expected_legs = ic_body_size // standard_leg_to_body_ratio

        # Display the estimated body size and expected number of legs
        cv2.putText(image, f"Body Size: {ic_body_size}, Expected Legs: {max_expected_legs}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Detect missing legs by comparing expected and detected legs
        missing_legs = max_expected_legs - number_of_legs

        # Normalize Missing Legs to Maximum Value
        if max_missing_legs < missing_legs:
            max_missing_legs = missing_legs

        # Display the frame with bounding boxes and leg count
        cv2.putText(image, f"Legs: {number_of_legs}, Missing: {missing_legs}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Frame with Bounding Boxes", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
