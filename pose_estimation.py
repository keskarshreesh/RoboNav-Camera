'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''


import numpy as np
import cv2
import time
from cv2 import aruco
import pickle
import os
import matplotlib.pyplot as plt

cam_extrinsics = np.array([
 [0.02136624,-0.99840068,0.05234092,-0.04847866],
 [-0.99970465,-0.0219419,-0.0104485,0.00539262],
 [0.01158024,-0.05210221,-0.99857461,0.07975163],
 [0.,0.,0.,1.]
])

# rvec_origin = np.array([0.00531156,2.99923678,-0.8294567])
# tvec_origin = np.array([0.02823084,-0.01628532,0.2125679])

trajectory = []
rotations = []
trajectory_box = []
rotations_box = []

x_converter = 8.95
y_converter = 12.17
x_offset = -0.04
y_offset = 0.1

sticker_points = [
                  [0,0],[0.237,0],[0.84,0],[-0.384,0],[-0.984,0],
                  [0.237,-0.60],[0.84,-0.60],[-0.384,-0.60],[-0.984,-0.60],
                  [0.237,-1.20],[0.84,-1.20],[-0.384,-1.20],[-0.984,-1.20],
                  [0.237,-1.80],[0.84,-1.80],[-0.384,-1.80],[-0.984,-1.80],
                 ]

# gt_lines = [
#             [[0,-0.384],[0,0]],
#             [[-0.384,-0.384],[0,-0.60]],
#             [[-0.384,0.237],[-0.60,-0.60]],
#             [[0.237,0.237],[-0.60,-1.20]],
#             [[0.237,-0.384],[-1.20,-1.20]],
#             [[-0.384,-0.384],[-1.20,-1.80]],
#             [[-0.384,0.237],[-1.80,-1.80]]
#            ]
# gt_lines = [
#             [[0,-0.384],[0,0]],
#             [[-0.384,0.237],[0,-0.60]],
#             [[0.237,-0.384],[-0.60,-1.20]],
#             [[-0.384,0.237],[-1.20,-1.80]]
#            ]
gt_lines = [
            [[0,0],[0,-1.80]]
           ]


def calculate_distance_from_line_segments(points, line_segments):
    distances = []

    for point in points:
        min_distance = np.inf

        for segment in line_segments:
            start, end = segment
            dist = np.linalg.norm(np.cross(np.array(end)-np.array(start), np.array(start)-point))/np.linalg.norm(np.array(end)-np.array(start))

            if dist < min_distance:
                min_distance = dist

        distances.append(min_distance)

    return np.array(distances)

def extract_rotations_from_matrix(matrix):
    # Extract individual rotations around x, y, and z axes
    x_rotation = np.arctan2(matrix[2, 1], matrix[2, 2])
    y_rotation = np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2))
    z_rotation = np.arctan2(matrix[1, 0], matrix[0, 0])

    # Convert radians to degrees
    x_rotation_degrees = np.degrees(x_rotation)
    y_rotation_degrees = np.degrees(y_rotation)
    z_rotation_degrees = np.degrees(z_rotation)

    return np.array([x_rotation_degrees, y_rotation_degrees, z_rotation_degrees])

def get_marker_world_coordinates(rvec,tvec):
    R_mark_to_cam, _ = cv2.Rodrigues(rvec)
    T_mark_to_cam = np.eye(4)
    T_mark_to_cam[:3,:3] = R_mark_to_cam
    T_mark_to_cam[:3,3] = tvec

    # if marker_id == 1:
    #     print(T_mark_to_cam)

    T_robot = (np.linalg.inv(cam_extrinsics) @ T_mark_to_cam)

    R_robot = T_robot[:3,:3]
    rvec_robot = extract_rotations_from_matrix(R_robot)
    Pw_robot = T_robot[:,3]

    # trajectory.append([Pw_robot[0]*x_converter,Pw_robot[1]*y_converter])
    # rotations.append(rvec_robot[2])
    return [[Pw_robot[0]*x_converter,Pw_robot[1]*y_converter],rvec_robot]

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    # parameters = cv2.aruco.DetectorParameters_create()

    dictionary = aruco.getPredefinedDictionary(aruco_dict_type)
    parameters =  aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)



    corners, ids, rejected_img_points = detector.detectMarkers(gray)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
        
            
            aruco.drawDetectedMarkers(frame, corners) 

            # Draw Axis
            # cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec_origin, tvec_origin, 0.01)
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  
            
            marker_coordinates, rotation_angles = get_marker_world_coordinates(rvec,tvec)

            # print('World coords of ref ' + str(ids[i]))
            # print(marker_coordinates)
            # print('Rotation of ref ' + str(ids[i]))
            # print(rotation_angles)

            if ids[i] == 1:
                trajectory.append(marker_coordinates)
                rotations.append(rotation_angles[2])
            elif ids[i] == 5:
                trajectory_box.append(marker_coordinates)
                rotations_box.append(rotation_angles[2])

            # Draw a square around the markers

    return frame

if __name__ == '__main__':

    aruco_dict_type = aruco.DICT_6X6_250
    
    
    with open('camera/matrices_1.pkl', 'rb') as f:
        data = pickle.load(f)
    k = data['cameraMatrix']
    d = data['distCoeffs']


    traj_dir = str(len(os.listdir('trajectories'))).zfill(3)
    traj_dir_path = os.path.join('trajectories',traj_dir)
    if not os.path.exists(traj_dir_path):
        os.makedirs(traj_dir_path)


    video = cv2.VideoCapture(0)
    if video.isOpened():
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        video.set(cv2.CAP_PROP_AUTOFOCUS, 0)  
        video.set(cv2.CAP_PROP_FPS, int(60))
    
    result = cv2.VideoWriter(os.path.join(traj_dir_path,'traj.avi'),cv2.VideoWriter_fourcc(*'XVID'),30.0,(1920,1080))

    time.sleep(2.0)
    
    s = time.time()

    result_frames = []

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        # result_no_pose.write(frame)
        
        output = pose_estimation(frame, aruco_dict_type, k, d)

        # result.write(output)
        result_frames.append(output)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        print(1/((time.time() - s)), (time.time() - s))
        s = time.time()

    for frame_num,frame in enumerate(result_frames):
        print(f"Writing video: Frame {str(frame_num)}")
        result.write(frame)

    trajectory = np.array(trajectory)
    rotations = np.array(rotations)
    trajectory[:,0] += x_offset
    trajectory[:,1] += y_offset
    with open(os.path.join(traj_dir_path,'traj.pkl'), 'wb') as f:
        pickle.dump({'trajectory': trajectory,'rotations': rotations}, f)

    if len(trajectory_box) > 0:
        trajectory_box = np.array(trajectory_box)
        rotations_box = np.array(rotations_box)
        trajectory_box[:,0] += x_offset
        trajectory_box[:,1] += y_offset
        with open(os.path.join(traj_dir_path,'traj_box.pkl'), 'wb') as f:
            pickle.dump({'trajectory_box': trajectory_box, 'rotations_box': rotations_box}, f)

    video.release()
    result.release()
    cv2.destroyAllWindows()

    plt.plot(trajectory[:,0],trajectory[:,1])
    for sticker_point in sticker_points:
        plt.plot(sticker_point[0],sticker_point[1],marker='o',color='green')

    plt.xlabel('X')
    plt.xlim(-2,2)
    plt.ylim(-3,1)
    plt.ylabel('Y')
    plt.title('Robot trajectory')
    plt.savefig(os.path.join(traj_dir_path,'traj_path.png'))
    plt.show()
    
    plt.scatter(np.arange(rotations.shape[0]),rotations)
    plt.ylabel('Rotation in degrees')
    plt.xlabel('iterations')
    plt.title('Robot rotation')
    plt.savefig(os.path.join(traj_dir_path,'rotation_degrees.png'))
    plt.show()

    ##Calculate metric
    # min_offsets_from_gt_traj = calculate_distance_from_line_segments(trajectory,gt_lines)
    # metric_frac_valid_points = np.sum(min_offsets_from_gt_traj <= 0.1)/len(trajectory)
    # print('Metric value:')
    # print(metric_frac_valid_points)