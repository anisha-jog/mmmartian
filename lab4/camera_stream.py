#! /usr/bin/env python3

# Streams the D435i head camera to a window for screen recording.
# Make sure to launch the camera first:
#   ros2 launch stretch_core d435i_low_resolution.launch.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraStream(Node):

    CAMERA_TOPIC = '/camera/color/image_raw'

    def __init__(self):
        super().__init__('camera_stream')
        self.bridge = CvBridge()
        self.create_subscription(Image, self.CAMERA_TOPIC, self._image_callback, 10)
        self.get_logger().info(f'Streaming {self.CAMERA_TOPIC} — press q to quit')

    def _image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow('Head Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                rclpy.shutdown()
        except Exception as e:
            self.get_logger().warn(f'Frame error: {e}')


if __name__ == '__main__':
    rclpy.init()
    node = CameraStream()
    rclpy.spin(node)
    cv2.destroyAllWindows()
