import lcm
from cam_lcm.cam_message_t import cam_message_t

msg = cam_message_t()
msg.timestamp = 0
msg.robot_pos = (0, 0, 0)
msg.obs_1_pos = (0, 0, 0)
msg.obs_2_pos = (0, 0, 0)
msg.obs_3_pos = (0, 0, 0)

lc = lcm.LCM()
lc.publish("CAM_POS", msg.encode())
