import cv2
import cv2.aruco as aruco
import numpy as np
import time

# Define the charuco board parameters
squaresX = 4  # Number of squares along the X-axis
squaresY = 6  # Number of squares along the Y-axis
squareLength = 6.75  # Length of each square in cm
markerLength = 6.1  # Length of each ArUco marker in cm
#dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)  # ArUco dictionary
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Create the charuco board object
# charuco_board = aruco.CharucoBoard_create(
#     squaresX, squaresY, squareLength, markerLength, dictionary
# )
charuco_board = aruco.CharucoBoard(
    size=(squaresX, squaresY), squareLength=squareLength, markerLength=markerLength, dictionary=dictionary
)
# charuco_board.setLegacyPattern(True)

arucoParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, arucoParams)

charucoParams = aruco.CharucoParameters()
charucoDetector = aruco.CharucoDetector(board=charuco_board,charucoParams=charucoParams,detectorParams=arucoParams)
# Capture calibration images
capture = cv2.VideoCapture(0)  # Adjust the camera index if needed
if capture.isOpened():
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
calibration_images = []
ctr = 0
while len(calibration_images) < 2000:  # Capture 20 calibration images
    ret, frame = capture.read()
    ctr +=  1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=arucoParams)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        # _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        #     corners, ids, gray, charuco_board
        # )
        charuco_corners, charuco_ids, _, _ = charucoDetector.detectBoard(
            gray,None,None,markerCorners=corners,markerIds=ids
        )

        if charuco_corners is not None and charuco_corners.shape[0] > 3:
            calibration_images.append(gray)
            aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    cv2.imshow("Charuco Calibration", frame)
    # time.sleep(2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

import random
random.shuffle(calibration_images)

# Perform camera calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
calibration_flags = cv2.CALIB_RATIONAL_MODEL
calibration_corners = []
calibration_ids = []
import random
for image in random.sample(calibration_images,120):
    corners, ids, rejected = aruco.detectMarkers(image, dictionary, parameters=arucoParams)
    if ids is not None and len(ids) > 0:
        _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, image, charuco_board
        )
        if charuco_corners is not None and charuco_corners.shape[0] > 3:
            calibration_corners.append(charuco_corners)
            calibration_ids.append(charuco_ids)

calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
    calibration_corners, calibration_ids, charuco_board,
    calibration_images[0].shape[::-1], None, None, criteria=criteria
)

# Print the camera matrix and distortion coefficients
import pickle
with open('camera/matrices_1.pkl', 'wb') as f:
    pickle.dump({'cameraMatrix': cameraMatrix, 'distCoeffs': distCoeffs}, f)
print("Camera Matrix:\n", cameraMatrix)
print("\nDistortion Coefficients:\n", distCoeffs)