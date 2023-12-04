#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np

theta1 = sp.symbols("theta1")
theta2 = sp.symbols("theta2")
theta3 = sp.symbols("theta3")

T = sp.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
a = [0, 0.08, 0.092]
d = [0.092, 0, 0]
alpha = [(sp.pi)/2, 0 , 0]
theta = [theta1 + sp.pi, theta2 + (sp.pi)/2, theta3]

for i in range(0,3):
  Ti = sp.Matrix([[sp.cos(theta[i]), -sp.sin(theta[i])*sp.cos(alpha[i]), sp.sin(theta[i])*sp.sin(alpha[i]), a[i]*sp.cos(theta[i])],
             [sp.sin(theta[i]), sp.cos(theta[i])*sp.cos(alpha[i]), -sp.cos(theta[i])*sp.sin(alpha[i]), a[i]*sp.sin(theta[i])],
             [0, sp.sin(alpha[i]), sp.cos(alpha[i]), d[i]],
             [0, 0, 0, 1]])
  T = T*Ti
  
sp.pprint(T)

'''Ininital spawn coordinates of the robot'''
START_X_COORDINATE = 0.0
START_Y_COORDINATE = 0.0

'''The goal coordinates for the robot'''
DESTINATION_X_COORDINATE = 5.0
DESTINATION_Y_COORDINATE = 6.0

Kp = 2.0                                               #proportional gain constant for steering control


class ToyCarController(Node):

    def __init__(self):                             
        self.wheel_angular_velocity = 0.0              # angular velocity of wheels in rad/s
        self.steering_angle = 0.0                      # steering angle in radians
        self.x = START_X_COORDINATE                    # the x coordiante of the robot
        self.y = START_Y_COORDINATE                    # the y coordinate of the robot
        self.x_vel = 0                                 # x component of linear velocity of robot
        self.y_vel = 0                                 # y component of linear velocity of robot
        self.yaw_final = 0.0                           # the required heading to the goal positon
        self.error_vs_t = []                           # list to store the values of error over time
        self.steering_angle_vs_t = []                  # list to store values of steering angle over time
        self.time = []                                 # list to store the current time
        self.n = 0                                     # variable to count the number of iterations
        self.wheel_velocities = Float64MultiArray()    # variable to store the angulary velocity of wheels command
        self.joint_positions = Float64MultiArray()     # variable to store the angulary position of steering command

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        super().__init__('sarb_controller')         # initialise toy_car_controller node
        self.subscription = self.create_subscription(Imu, '/imu_plugin/out', self.navigation_callback, qos_profile)
        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.wheel_velocities_pub = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        

    '''navigation_callback function processes the imu data to find the current heading, then calulates the required heading to
    the goal position. Using these two values, it calculates error and gives the steering commands using a P controller'''
    def navigation_callback(self, msg):
        # extract quaternion from imu message data
        x = msg.orientation.x  
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        self.yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))                    # convert quaternion to yaw angle
        self.x_vel = (self.wheel_angular_velocity*0.025)*math.cos(self.yaw) * 0.70         # calculate the x component of linear velocity
        self.y_vel = (self.wheel_angular_velocity*0.025)*math.sin(self.yaw) * 0.715        # calculate the y component of linear velocity
        self.x += self.x_vel*(1/100)                                                       # calculate the dispacement of robot along x axis
        self.y += self.y_vel*(1/100)                                                       # calculate the dispacement of robot along y axis
        
        self.yaw_final = math.atan2(DESTINATION_Y_COORDINATE-self.y, DESTINATION_X_COORDINATE-self.x)

        if ((DESTINATION_X_COORDINATE-0.15) <= self.x <= (DESTINATION_X_COORDINATE+0.15)) and (
            (DESTINATION_Y_COORDINATE-0.15) <= self.y <= (DESTINATION_Y_COORDINATE+0.15)):          #check if the root has reached the desired goal coordinates
            self.get_logger().info("Reached", once=True)
            self.wheel_angular_velocity = 0.                                                        #set angular velocity of wheels to 0 and stop the robot
            self.steering_angler = 0.0
            self.publish_commands()
            # self.plot_trajectory_data()
        else:
            self.wheel_angular_velocity = 20.0                                                      
            self.error = self.yaw_final - self.yaw                                                  # calculate the error between current heading and desired heading
            self.steering_angle = Kp*self.error                                                     # calculate the steering angle using a P controller
            self.get_logger().info(f'X: {round(self.x, 2)} , Y: {round(self.y, 2)}')
            self.get_logger().info(f'Yaw error: {round(self.error, 2)} , Steering angle: {round(self.steering_angle, 2)}')
            self.publish_commands()

        self.n += 1
        self.error_vs_t.append(self.error)                             # store the heading error over time
        self.steering_angle_vs_t.append(self.steering_angle)           # store the steering angle over time
        self.time.append(self.n/100)                                   # store respective time stamps


    '''publish_commands function publishes the commands to /position_controller/commands and /velocity_controller/commands
    topics'''
    def publish_commands(self):
        self.joint_positions.data = [-self.steering_angle, -self.steering_angle,0.0, 0.0, 0.0]
        self.wheel_velocities.data = [self.wheel_angular_velocity,-self.wheel_angular_velocity]
        self.wheel_velocities_pub.publish(self.wheel_velocities)
        self.joint_position_pub.publish(self.joint_positions)


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

