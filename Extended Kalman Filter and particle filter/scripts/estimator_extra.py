#!/usr/bin/env python  
import rospy
import math
import numpy
import scipy.stats
from state_estimator.msg import SensorData
from geometry_msgs.msg import Pose2D

class Extimator(object):
    def __init__(self):
        self.F = numpy.array(numpy.eye(3))
	self.x = 0
	self.y = 0
	self.theta = 0
        self.x_true = numpy.array([[0],[0],[0]])

        
        self.V = numpy.array([[0.1, 0, 0],
                              [0, 0.1, 0],
			      [0, 0, 0.1]])
        
        
	self.P_est = numpy.array([[0,0,0],[0,0,0],[0,0,0]])
        
	self.particles = self.create_gaussian_particles(0, 0.1, 500)
	
	self.weight = numpy.ones(500)
	rospy.Subscriber("sensor_data", SensorData, self.callback)
	rospy.Subscriber("/robot_pose", Pose2D, self.callback_1)
	self.pub_pose = rospy.Publisher("/robot_pose_estimate", Pose2D, queue_size=10)
	self.pub_pose_1 = rospy.Publisher("/robot_pose_estimate_1", Pose2D, queue_size=10)

    def create_gaussian_particles(self, mean, covariance, N):
        particles = numpy.empty((N, 3))
     	particles[:, 0] = mean + (numpy.random.randn(N) * covariance)
     	particles[:, 1] = mean + (numpy.random.randn(N) * covariance)
     	particles[:, 2] = mean + (numpy.random.randn(N) * covariance)
     	
     	return particles
    
    def callback_1(self, data):
	self.x = data.x
	self.y = data.y
	self.theta = data.theta

    def callback(self, data):
	self.EFK(data)
	self.PF(data)

    def EFK(self, data):
	v_trans = data.vel_trans
	v_ang = data.vel_ang
	self.F[0][2] = -0.01 * v_trans * numpy.sin(self.x_true[2])
	self.F[1][2] = 0.01 * v_trans * numpy.cos(self.x_true[2])
	x_hat = numpy.empty((3, 1))
	x_hat[0] = self.x_true[0] + 0.01 * v_trans * numpy.cos(self.x_true[2])
	x_hat[1] = self.x_true[1] + 0.01 * v_trans * numpy.sin(self.x_true[2])
	x_hat[2] = self.x_true[2] + 0.01 * v_ang
	P_hat = numpy.dot(numpy.dot(self.F, self.P_est), numpy.transpose(self.F)) + self.V
	n = len(data.readings)
	if n == 0:
	    self.x_true = x_hat
	    self.P_est = P_hat
	else:
	    H = numpy.zeros((2*n, 3))
	    Hx = numpy.zeros((2*n, 1))
	    y = numpy.empty((2*n, 1))
	    S = numpy.zeros((2*n, 2*n))
	    for i in range(0, n):
		xl = data.readings[i].landmark.x
		yl = data.readings[i].landmark.y
		H[2*i][0] = (x_hat[0] - xl) / numpy.sqrt((x_hat[0] - xl)**2 + (x_hat[1] - yl)**2)
		H[2*i][1] = (x_hat[1] - yl) / numpy.sqrt((x_hat[0] - xl)**2 + (x_hat[1] - yl)**2)
		H[2*i + 1][0] = (yl - x_hat[1]) / ((x_hat[0] - xl)**2 + (x_hat[1] - yl)**2)
		H[2*i + 1][1] = (x_hat[0] - xl) / ((x_hat[0] - xl)**2 + (x_hat[1] - yl)**2)
		H[2*i + 1][2] = -1
		y[2*i] = data.readings[i].range
		y[2*i+1] = data.readings[i].bearing
		Hx[2*i] = math.sqrt((x_hat[0]-xl)**2 + (x_hat[1]-yl)**2) 
		Hx[2*i+1] = math.atan2((yl - x_hat[1]), (xl - x_hat[0])) - x_hat[2] 
	    W = 0.1 * numpy.eye(2 * n)	
	    mu = numpy.subtract(y, Hx)
	    S = numpy.dot(numpy.dot(H, P_hat), numpy.transpose(H)) + W
	    R = numpy.dot(numpy.dot(P_hat, numpy.transpose(H)), numpy.linalg.inv(S))
	    self.x_true = x_hat + numpy.dot(R, mu)
	    self.P_est = numpy.subtract(P_hat, numpy.dot(numpy.dot(R, H), P_hat))
	
	msg = Pose2D()
        msg.x = self.x_true[0]
        msg.y = self.x_true[1]
        msg.theta = self.x_true[2]
        self.pub_pose.publish(msg)

    def PF(self, data):
	m = self.particles.shape[0]
	v_trans = data.vel_trans
	v_ang = data.vel_ang
	self.particles[:, 0] = self.particles[:, 0] + (0.01 * (v_trans + numpy.random.randn(m) * 0.1)) * numpy.cos(self.particles[:, 2])
	self.particles[:, 1] = self.particles[:, 1] + (0.01 * (v_trans + numpy.random.randn(m) * 0.1)) * numpy.sin(self.particles[:, 2])
	self.particles[:, 2] = self.particles[:, 2] + 0.01 * (v_ang + numpy.random.randn(m) * 0.1)
	
	n = len(data.readings)
	if n > 0:
	    W = numpy.random.normal(0, 0.1, (m,3))
	    self.particles = self.particles + W
	    landmarks = numpy.empty((n, 2))
	    y = numpy.empty((n, 2))
	    self.weight = numpy.ones(m)
	    for i in range(0, n):
		landmarks[i][0] = data.readings[i].landmark.x 
		landmarks[i][1] = data.readings[i].landmark.y
		y[i][0] = data.readings[i].range 
		y[i][1] = data.readings[i].bearing 
	    
	    weight_1 = numpy.ones(m)
	    weight_2 = numpy.ones(m)
	    
	    distance = numpy.empty(m)
	    bearing_e = numpy.zeros(m)
	    for i, landmark in enumerate(landmarks):
        	distance = numpy.linalg.norm(self.particles[:, 0:2] - landmark, axis=1) 
        	weight_1 *= scipy.stats.norm(distance, 0.1).pdf(y[i][0])
		
	    for i in range(n):
		for j in range(m):
		    bearing_e[j] = math.atan2((landmarks[i, 1]-self.particles[j, 1]), (landmarks[i, 0]-self.particles[j, 0])) - self.particles[j, 2]
		weight_2 *= scipy.stats.norm(bearing_e, 0.1).pdf(y[i][1])
	    self.weight = abs(weight_1) * abs(weight_2)
	    self.weight += 1.e-300
	    self.weight /= sum(self.weight)

	    csum = numpy.cumsum(self.weight)
 	    csum[-1] = 1
	    index = numpy.searchsorted(csum,  numpy.random.rand(m))
	    self.weight[:] = self.weight[index]
	    self.weight /= numpy.sum(self.weight)
	    self.particles[:] = self.particles[index]

	position = numpy.average(self.particles, weights = self.weight, axis = 0)
	ee = numpy.sqrt((self.x_true[0] - self.x)**2 + (self.x_true[1] - self.y)**2 + (self.x_true[2] - self.theta)**2)
	ep = numpy.sqrt((position[0] - self.x)**2 + (position[1] - self.y)**2 + (position[2] - self.theta)**2)

	print("the error of EFK:",ee, "the error of PF:",ep)
	
	msg = Pose2D()
        msg.x = position[0]
        msg.y = position[1]
        msg.theta = position[2]
        self.pub_pose_1.publish(msg)


if __name__ == '__main__':
    estimator = Extimator()
    rospy.init_node('estimator', anonymous=True)
    rospy.spin()
