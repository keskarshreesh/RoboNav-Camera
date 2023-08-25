import cv2
import cv2.aruco as aruco

# Define the charuco board parameters
squaresX = 4  # Number of squares along the X-axis
squaresY = 6  # Number of squares along the Y-axis
squareLength = 200  # Length of each square in pixels
markerLength = 180  # Length of each ArUco marker in pixels
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # ArUco dictionary

# Create the charuco board object
charuco_board = aruco.CharucoBoard(
    (squaresX, squaresY), squareLength, markerLength, dictionary
)

# Define the size of the charuco board image
imageSize = (810,1215)

# Generate the charuco board image
board_image = charuco_board.generateImage((squaresX*squareLength,squaresY*squareLength))

# Save the charuco board image
cv2.imwrite("charuco/charuco_board_2.png", board_image)

# Display the charuco board image
cv2.imshow("Charuco Board", board_image)
cv2.waitKey(0)
cv2.destroyAllWindows()