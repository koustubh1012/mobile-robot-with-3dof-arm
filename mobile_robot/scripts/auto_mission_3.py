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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import PoseStamped

theta1, theta2, theta3 = sp.symbols('theta1, theta2, theta3')

class ToyCarController(Node):

    def homo_matrix(self,a,d,alpha,theta):                                                                                                                 #D-H Table function for calculating the homogeneous matrices

        T = sp.Matrix([[sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha), a * sp.cos(theta)],
                    [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a * sp.sin(theta)],
                    [0, sp.sin(alpha), sp.cos(alpha), d],
                    [0, 0, 0, 1]])
        return T

    def __init__(self):        
        T_1 = self.homo_matrix(0, 0.092, (sp.pi / 2), theta1 + sp.pi)
        T_2 = self.homo_matrix(0.08, 0, 0, theta2 + (sp.pi/2))
        T_3 = self.homo_matrix(0.092,0,0,theta3)
        T_01 = T_1
        T_02 = T_1 * T_2
        T_03 = T_1 * T_2 * T_3
        J_w = sp.Matrix([[T_01[0,2],T_02[0,2],T_03[0,2]],[T_01[1,2],T_02[1,2],T_03[1,2]],[T_01[2,2],T_02[2,2],T_03[2,2]]])
        P_x = T_03[0,3]
        P_y = T_03[1,3]
        P_z = T_03[2,3]
        J_v = sp.Matrix([[0,0,0],[0,0,0],[0,0,0]])
        J_v[0,0] = sp.diff(P_x, theta1)
        J_v[1,0] = sp.diff(P_y, theta1)
        J_v[2,0] = sp.diff(P_z, theta1)
        J_v[0,1] = sp.diff(P_x, theta2)
        J_v[1,1] = sp.diff(P_y, theta2)
        J_v[2,1] = sp.diff(P_z, theta2)
        J_v[0,2] = sp.diff(P_x, theta3)
        J_v[1,2] = sp.diff(P_y, theta3)
        J_v[2,2] = sp.diff(P_z, theta3)
        self.J = sp.Matrix.vstack(J_v, J_w)
        theta_values = {theta1: 0.00001, theta2: 0.000001, theta3: 0}
        self.J_sp = self.J.subs(theta_values)
        J_value = np.matrix(self.J_sp).astype(np.float64)
        self.J_inv = np.linalg.pinv(J_value)
        self.Time = 12            # total time to draw the circle
        self.t = 0
        self.dt = 0.1
        self.q = np.zeros((3,1))
        self.X_values = []
        self.Y_values = []
        self.Z_values = []
        self.z_dot = -0.0092
        self.x_dot = 0.0092
        self.y_dot = 0
        self.t_t = []
        self.X_dot = np.matrix([[self.x_dot], [self.y_dot], [self.z_dot], [0], [0], [0]])

        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0                           
        self.joint_positions = Float64MultiArray()     # variable to store the angulary position of steering command
        self.wheel_velocities = Float64MultiArray() 

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        super().__init__('sarb_controller')         # initialise toy_car_controller node
        self.timer = self.create_timer(0.1, self.timer_callback)  # Creating a timer that triggers every 1 second
        self.subscription = self.create_subscription(PoseStamped, '/odom', self.odom_callback, qos_profile)
        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.wheel_velocities_pub = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

    def timer_callback(self):
        if (self.t < self.Time):
            q_dot = self.J_inv*self.X_dot
            self.q = self.q + q_dot*self.dt
            [angle1, angle2, angle3] = [self.q[i].item() for i in range(3)]
            self.J_sp = self.J.subs({theta1: angle1, theta2: angle2, theta3: angle3})
            J_value = np.matrix(self.J_sp).astype(np.float64)
            self.J_inv = np.linalg.pinv(J_value)
            self.t = self.t + self.dt
            self.q1 = self.q[0,0]
            self.q2 = self.q[1,0]
            self.q3 = self.q[2,0]
            self.publish_commands()
        else:
            self.get_logger().info("Reached", once="True")
            self.plot_trajectory_data()


    def odom_callback(self,msg):
        self.X_values.append(msg.pose.position.x)
        self.Y_values.append(msg.pose.position.y)
        self.Z_values.append(msg.pose.position.z)
        # if (self.q2 <= -1.57):
        #     self.q2 = -1.57
        #     self.q1 = 1.57
        #     self.publish_commands()
        # else:
        #     self.q2 -= 0.01
        #     self.q1 += 0.01
        #     self.publish_commands()


    '''publish_commands function publishes the commands to /position_controller/commands and /velocity_controller/commands
    topics'''
    def publish_commands(self):
        self.joint_positions.data = [0.0,0.0,-self.q1, self.q2,self.q3]
        self.wheel_velocities.data = [0.0, 0.0]
        self.wheel_velocities_pub.publish(self.wheel_velocities)
        self.joint_position_pub.publish(self.joint_positions)
        # self.get_logger().info(self.joint_positions.data)


    '''plot_trajectory_data plots the heading error and the steering angle control over time'''
    def plot_trajectory_data(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X_values, self.Y_values, self.Z_values)
        ax.set_xlabel('X coordinates')
        ax.set_ylabel('Y coordinates')
        ax.set_zlabel('Z coordinates')
        ax.set_title('Trajectory of the robot')
        plt.show()




def main(args=None):
    rclpy.init(args=args)
    controller = ToyCarController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
