import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Start webcam

while True:
    _, frame = cap.read()
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for blue color in HSV
    lower_blue = np.array([60, 35, 140])
    upper_blue = np.array([180, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Filter the blue region
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Show frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Blue Mask', mask)
    cv2.imshow('Blue Filtered Result', result)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop when 'q' is pressed

cap.release()
cv2.destroyAllWindows()
print("Exiting...")  # Confirm exit