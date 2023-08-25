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
import lcm
from cam_lcm.cam_message_t import cam_message_t

cam_extrinsics = np.array([
 [0.02136624,-0.99840068,0.05234092,-0.04847866],
 [-0.99970465,-0.0219419,-0.0104485,0.00539262],
 [0.01158024,-0.05210221,-0.99857461,0.07975163],
 [0.,0.,0.,1.]
])

x_converter = 8.95
y_converter = 12.17
x_offset = -0.04
y_offset = 0.1

object_to_marker_id_map = {
    "robot": 1,
    "obs_1": 5,
    "obs_2": 6,
    "obs_3": 7
}

marker_id_to_pos_map = {
    1: (0,0,0),
    5: (0,0,0),
    6: (0,0,0),
    7: (0,0,0)
}

cam_lcm = lcm.LCM()


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

    return np.array([x_rotation, y_rotation, z_rotation])

def get_marker_world_coordinates(rvec,tvec):
    R_mark_to_cam, _ = cv2.Rodrigues(rvec)
    T_mark_to_cam = np.eye(4)
    T_mark_to_cam[:3,:3] = R_mark_to_cam
    T_mark_to_cam[:3,3] = tvec

    T_robot = (np.linalg.inv(cam_extrinsics) @ T_mark_to_cam)

    R_robot = T_robot[:3,:3]
    rvec_robot = extract_rotations_from_matrix(R_robot)
    Pw_robot = T_robot[:,3]

    return [[(Pw_robot[0]*x_converter) + x_offset,(Pw_robot[1]*y_converter) + y_offset],rvec_robot]

def publish_cam_msg():
    msg = cam_message_t()
    msg.timestamp = int(time.time())
    msg.robot_pos = marker_id_to_pos_map[object_to_marker_id_map["robot"]]
    msg.obs_1_pos = marker_id_to_pos_map[object_to_marker_id_map["obs_1"]]
    msg.obs_2_pos = marker_id_to_pos_map[object_to_marker_id_map["obs_2"]]
    msg.obs_3_pos = marker_id_to_pos_map[object_to_marker_id_map["obs_3"]]
    cam_lcm.publish("CAM_POS", msg.encode())

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  
            
            marker_coordinates, rotation_angles = get_marker_world_coordinates(rvec,tvec)

            # print('World coords of ref ' + str(ids[i]))
            # print(marker_coordinates)
            # print('Rotation of ref ' + str(ids[i]))
            # print(rotation_angles)

            marker_id_to_pos_map[ids[i][0]] = (marker_coordinates[0],marker_coordinates[1],rotation_angles[2])
            publish_cam_msg()

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

    # result_frames = []

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_estimation(frame, aruco_dict_type, k, d)
        # result_frames.append(output)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        print(1/((time.time() - s)), (time.time() - s))
        s = time.time()

    # for frame_num,frame in enumerate(result_frames):
    #     print(f"Writing video: Frame {str(frame_num)}")
    #     result.write(frame)

    video.release()
    result.release()
    cv2.destroyAllWindows()