#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import matplotlib.pyplot as plt


class ToyCarController(Node):

    def __init__(self):
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0                           
        self.joint_positions = Float64MultiArray()     # variable to store the angulary position of steering command
        self.wheel_velocities = Float64MultiArray() 

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        super().__init__('sarb_controller')         # initialise toy_car_controller node
        self.subscription = self.create_subscription(Imu, '/imu_plugin/out', self.navigation_callback, qos_profile)
        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.wheel_velocities_pub = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        

    '''navigation_callback function processes the imu data to find the current heading, then calulates the required heading to
    the goal position. Using these two values, it calculates error and gives the steering commands using a P controller'''
    def navigation_callback(self, msg):
        if (self.q2 <= -1.57):
            self.q2 = -1.57
            self.q1 = 1.57
            self.publish_commands()
        else:
            self.q2 -= 0.01
            self.q1 += 0.01
            self.publish_commands()


    '''publish_commands function publishes the commands to /position_controller/commands and /velocity_controller/commands
    topics'''
    def publish_commands(self):
        self.joint_positions.data = [0.0,0.0,self.q1,self.q2,0.0]
        self.wheel_velocities.data = [0.0, 0.0]
        self.wheel_velocities_pub.publish(self.wheel_velocities)
        self.joint_position_pub.publish(self.joint_positions)
        # self.get_logger().info(self.joint_positions.data)


    '''plot_trajectory_data plots the heading error and the steering angle control over time'''
    def plot_trajectory_data(self):
        plt.plot(self.time, self.error_vs_t, label='Yaw error')
        plt.plot(self.time, self.steering_angle_vs_t, label='Steering angle')
        plt.ylabel("Angle in radians")
        plt.xlabel("Time in seconds")
        plt.legend()
        plt.show()




def main(args=None):
    rclpy.init(args=args)
    controller = ToyCarController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

