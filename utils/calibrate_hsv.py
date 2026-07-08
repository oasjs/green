import cv2
import numpy as np

def calibrate_hsv(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("Mask")
    cv2.createTrackbar("H min", "Mask",  0, 179, lambda x: None)
    cv2.createTrackbar("H max", "Mask", 179, 179, lambda x: None)
    cv2.createTrackbar("S min", "Mask",  0, 255, lambda x: None)
    cv2.createTrackbar("S max", "Mask", 255, 255, lambda x: None)
    cv2.createTrackbar("V min", "Mask",  0, 255, lambda x: None)
    cv2.createTrackbar("V max", "Mask", 255, 255, lambda x: None)

    while True:
        h_min = cv2.getTrackbarPos("H min", "Mask")
        h_max = cv2.getTrackbarPos("H max", "Mask")
        s_min = cv2.getTrackbarPos("S min", "Mask")
        s_max = cv2.getTrackbarPos("S max", "Mask")
        v_min = cv2.getTrackbarPos("V min", "Mask")
        v_max = cv2.getTrackbarPos("V max", "Mask")

        low  = np.array([h_min, s_min, v_min])
        high = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, low, high)

        result = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"LOW  = {low}")
            print(f"HIGH = {high}")
            break

    cv2.destroyAllWindows()

calibrate_hsv("sua_imagem.jpg")