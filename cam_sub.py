import lcm
from cam_lcm.cam_message_t import cam_message_t

def my_handler(channel, data):
    msg = cam_message_t.decode(data)
    print("Received message on channel \"%s\"" % channel)
    print("   timestamp   = %s" % str(msg.timestamp))
    print("   robot_pos   = %s" % str(msg.robot_pos))
    print("   Obstacle 1 Position = %s" % str(msg.obs_1_pos))
    print("   Obstacle 2 Position = %s" % str(msg.obs_2_pos))
    print("   Obstacle 3 Position = %s" % str(msg.obs_3_pos))
    print("")

lc = lcm.LCM()
subscription = lc.subscribe("CAM_POS", my_handler)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass