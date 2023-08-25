import cv2
import cv2.aruco as aruco

for i in range(10):
    # Define the charuco board parameters
    squaresX = 1  # Number of squares along the X-axis
    squaresY = 1  # Number of squares along the Y-axis
    squareLength = 200  # Length of each square in pixels
    markerLength = 180  # Length of each ArUco marker in pixels
    # dictionary = aruco.Dictionary_get(aruco.DICT_6X6_250)  # ArUco dictionary
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # ArUco dictionary

    # Define the size of the charuco board image
    img = aruco.drawMarker(dictionary, i, 700)
    cv2.imwrite(f"aruco/{i}.jpg", img)

# Display the image to us
# cv2.imshow('frame', img)
# Exit on any key
# cv2.waitKey(0)
# cv2.destroyAllWindows()