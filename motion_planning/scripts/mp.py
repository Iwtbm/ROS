#!/usr/bin/env python  
import rospy
import math
import numpy
from urdf_parser_py.urdf import URDF
from geometry_msgs.msg import Transform
from cartesian_control.msg import CartesianCommand
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
import tf
import tf2_ros
import geometry_msgs.msg
import sys
import moveit_msgs.msg
import moveit_msgs.srv
import moveit_commander
import trajectory_msgs

def convert_to_message(T):
    t = geometry_msgs.msg.Pose()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[0]
    t.orientation.y = orientation[1]
    t.orientation.z = orientation[2]
    t.orientation.w = orientation[3]        
    return t

class Node:
    def __init__(self, value, parent=None):  
        self.value = value  
        self.parent = parent

class MoveArm(object):

    def __init__(self):
        self.robot = URDF.from_parameter_server()
        
        self.num_joints = 0
        self.joint_names = []
        self.joint_axes = []
	self.joint_transforms = []
        self.x_current = tf.transformations.identity_matrix()
        self.get_joint_info()
        self.current_joint_state = JointState()
        
	rospy.Subscriber("joint_states", JointState, self.js_callback)

        # Wait for validity check service
        rospy.wait_for_service("check_state_validity")
        self.state_valid_service = rospy.ServiceProxy('check_state_validity',  
                                                      moveit_msgs.srv.GetStateValidity)
        print "State validity service ready"

        # MoveIt parameter
        self.group_name = "lwr_arm"
        
        rospy.Subscriber("/motion_planning_goal", Transform, self.mp_callback)

	self.pub_trajectory = rospy.Publisher("/joint_trajectory", JointTrajectory, queue_size=1)

    def get_joint_info(self):
        self.num_joints = 0
        self.joint_names = []
        self.joint_axes = []
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break            
            (joint_name, next_link) = self.robot.child_map[link][0]
            current_joint = self.robot.joint_map[joint_name]        
            if current_joint.type != 'fixed':
                self.num_joints = self.num_joints + 1
                self.joint_names.append(current_joint.name)
                self.joint_axes.append(current_joint.axis)
            link = next_link

    def process_link_recursive(self, link, T, joint_values, qc, judge):
        if link not in self.robot.child_map: 
            self.x_current = T
            return
        for i in range(0,len(self.robot.child_map[link])):
            (joint_name, next_link) = self.robot.child_map[link][i]
            if joint_name not in self.robot.joint_map:
                rospy.logerror("Joint not found in map")
                continue
            current_joint = self.robot.joint_map[joint_name]        

            trans_matrix = tf.transformations.translation_matrix((current_joint.origin.xyz[0], 
                                                                  current_joint.origin.xyz[1],
                                                                  current_joint.origin.xyz[2]))
            rot_matrix = tf.transformations.euler_matrix(current_joint.origin.rpy[0], 
                                                         current_joint.origin.rpy[1],
                                                         current_joint.origin.rpy[2], 'rxyz')
            origin_T = numpy.dot(trans_matrix, rot_matrix)
            current_joint_T = numpy.dot(T, origin_T)
            if current_joint.type != 'fixed':
                if current_joint.name not in joint_values.name:
                    rospy.logerror("Joint not found in list")
                    continue
                self.joint_transforms.append(current_joint_T)
                index = joint_values.name.index(current_joint.name)
		if judge == True:
                    angle = joint_values.position[index]
                else:
                    angle = qc[index]
                joint_rot_T = tf.transformations.rotation_matrix(angle, 
                                                                 numpy.asarray(current_joint.axis))
                next_link_T = numpy.dot(current_joint_T, joint_rot_T) 
            else:
                next_link_T = current_joint_T

            self.process_link_recursive(next_link, next_link_T, joint_values, qc, judge)
  
    def rotation_from_matrix(self, matrix):
        R = numpy.array(matrix, dtype=numpy.float64, copy=False)
        R33 = R[:3, :3]
        # axis: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, W = numpy.linalg.eig(R33.T)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        axis = numpy.real(W[:, i[-1]]).squeeze()
        # point: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, Q = numpy.linalg.eig(R)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        # rotation angle depending on axis
        cosa = (numpy.trace(R33) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
        angle = math.atan2(sina, cosa)
        return angle, axis
    
    def position_from_matrix(self, matrix):
        angle, axis = self.rotation_from_matrix(matrix)
        rotation_change = numpy.dot(angle, axis)
        rotation_change = rotation_change.reshape(3,1)
        translation_change = tf.transformations.translation_from_matrix(matrix)
        translation_change = translation_change.reshape(3,1)
        x = numpy.row_stack((translation_change, rotation_change))
        return x

    def inverse_kinematics(self, T_c, T_d, joint_transforms):
        v = []
        V_j = []
        iT_c = tf.transformations.inverse_matrix(T_c)
        delta_x = numpy.dot(iT_c, T_d)
        x_dot = self.position_from_matrix(delta_x)
        for jt in joint_transforms:
            jTee = numpy.dot(tf.transformations.inverse_matrix(jt), T_c)
            eeTj = numpy.dot(iT_c, jt)
            Sj = [[0, -jTee[2, 3], jTee[1, 3]], [jTee[2, 3], 0, -jTee[0, 3]], [-jTee[1, 3], jTee[0, 3], 0]]
            V_j.append(numpy.row_stack((numpy.column_stack((numpy.zeros((3,3)), -numpy.dot(eeTj[:3, :3], Sj))), numpy.column_stack((numpy.zeros((3,3)), eeTj[:3, :3])))))
        for i in range(0, len(self.joint_axes)):
            for j in range(0, len(self.joint_axes[i])):
                if self.joint_axes[i][j] != 0:
                    v.append(V_j[i][:, j+3]*self.joint_axes[i][j])	            
        J = v[0].reshape(6, 1)
        for i in range(1, len(v)):
            J = numpy.column_stack((J, v[i].reshape(6, 1)))
        Jp = numpy.linalg.pinv(J)
        return (Jp, x_dot)

    def IK(self, T_xd):
        n = len(self.joint_names)
	q = []
        qc = numpy.zeros((n, 1))
        for i in range(0, n):
            qc[i] = numpy.random.uniform(-numpy.pi, numpy.pi)
        delta_q = numpy.ones((n, 1))
        dx = 1
        j = 0
        k = 0
	root = self.robot.get_root()
        while dx > 0.01:
            absdx = []
            absdth = []
            T = tf.transformations.identity_matrix()
            self.joint_transforms = []
            self.process_link_recursive(root, T, self.current_joint_state, qc, judge = False)
            (Jp, x_dot) = self.inverse_kinematics(self.x_current, T_xd, self.joint_transforms)
            delta_q = numpy.dot(Jp, x_dot)
            dx = numpy.amax(numpy.abs(x_dot))
            for i in range(0, n):
                index = self.current_joint_state.name.index(self.joint_names[i])
                qc[index] = qc[index] + delta_q[i]
            j = j + 1
            if j > 600:
                for i in range(0, n):
                    qc[i] = numpy.random.uniform(-numpy.pi, numpy.pi)
                j = 0
                k = k+1
            if k > 30:
                raise ValueError('Cannot reach that point!')
	qc = qc.reshape(1,n)
        for i in range(0, n):
            index = self.current_joint_state.name.index(self.joint_names[i])
            q.append(qc[0][index])
	for i in range(0, n):
	    while q[i] > numpy.pi:
		q[i] = q[i] - 2 * numpy.pi
	    while q[i] < -numpy.pi:
		q[i] = q[i] + 2 * numpy.pi	
	if self.is_state_valid(q) == True:
	    return q
	else:
	    q = []
	    return q

    """ This function checks if a set of joint angles q[] creates a valid state, or 
    one that is free of collisions. The values in q[] are assumed to be values for 
    the joints of the KUKA arm, ordered from proximal to distal. 
    """
    def is_state_valid(self, q):
	p = []
	req = moveit_msgs.srv.GetStateValidityRequest()
        req.group_name = self.group_name
        req.robot_state = moveit_msgs.msg.RobotState()
	n = len(self.joint_names)
	req.robot_state.joint_state.name = self.current_joint_state.name
	for i in range(len(self.joint_names)):
	    #j = self.current_joint_state.name.index(self.joint_names[i])
	    j = self.joint_names.index(self.current_joint_state.name[i])
	    p.append(q[j])
	req.robot_state.joint_state.position = p
	req.robot_state.joint_state.velocity = numpy.zeros(n)
        req.robot_state.joint_state.effort = numpy.zeros(n)
        req.robot_state.joint_state.header.stamp = rospy.get_rostime()
        res = self.state_valid_service(req)
        return res.valid

    def js_callback(self, joint_values): 
	self.current_joint_state = joint_values

    def T_desire(self, x, y, z, rx, ry, rz, w):
    	T_d = tf.transformations.concatenate_matrices(
		tf.transformations.translation_matrix((x, y, z)),
		tf.transformations.quaternion_matrix((rx, ry, rz, w))
        	)
    	return T_d 

    def partition_path(self, p_closest, q, segment = 0.1):
	T = []
	num_parts = max(numpy.ceil(numpy.true_divide(abs(numpy.subtract(p_closest, q)), segment)))
	a = numpy.true_divide(numpy.subtract(q, p_closest), num_parts)
	for i in range(int(num_parts)+1):
	    T.append(a*i+p_closest)
	return T

    def judge(self, p_closest, q):
	T = self.partition_path(p_closest, q)
	for t in T:
	    if self.is_state_valid(t) == False:
		return False
	return True 

    def motion_plannning(self, q_s, q_d):
	tree = []
	q_inverse = []
        q_listb = []
	q_list = []
	tree.append(Node(q_s))
	begin = rospy.get_rostime().secs
	while self.judge(tree[len(tree)-1].value, q_d) == False:
	    q = []
	    d = []
	    for i in range(len(self.joint_names)):
	    	q.append(numpy.random.uniform(-numpy.pi, numpy.pi))
	    for i in range(len(tree)):
	    	d.append(numpy.linalg.norm(numpy.subtract(tree[i].value, q)))
	    j = d.index(min(d))
	    p_closest = tree[j].value 
	    delta = numpy.subtract(q, p_closest)
	    q_c = numpy.add(numpy.multiply(0.5 / numpy.linalg.norm(delta), delta) , p_closest)
	    if self.judge(p_closest, q_c) == True:
		a = Node(q_c)
		a.parent = tree[j]
		tree.append(a)
	now = rospy.get_rostime().secs
	print("Using time: %d s" %(now - begin))
	a = Node(q_d)
	a.parent = tree[len(tree)-1]
	tree.append(a)	
        while True:
	    q_inverse.append(a.value)
	    if a.parent == None:
		break
	    else:
		a = a.parent
	q_inverse.reverse()
	q_lista = q_inverse
	q_listb.append(q_s)
	k = 0
	for i in range(len(q_lista)-1):
	    j = i+1
	    if (numpy.subtract(q_lista[i], q_lista[k])).any() != False:
		continue
	    while self.judge(q_lista[i], q_lista[j]) == True:
		j = j + 1
		if j == len(q_lista):
		    break
	    q_listb.append(q_lista[j-1])
	    k = j-1
	for i in range(len(q_listb)-1):
	    if numpy.linalg.norm(numpy.subtract(q_listb[i], q_listb[i+1])) > 0.5:
		T = []
		num_parts = max(numpy.ceil(numpy.true_divide(abs(numpy.subtract(q_listb[i+1], q_listb[i])), 0.5)))
		a = numpy.true_divide(numpy.subtract(q_listb[i+1], q_listb[i]), num_parts)
		for j in range(int(num_parts)+1):
	    	    T.append(a*j+q_listb[i])
		for k in range(len(T)-1):
		    q_list.append(T[k])
	    else:
		q_list.append(q_listb[i])
	q_list.append(q_listb[-1])
	return q_list

    def path(self, q_list):
	path = JointTrajectory()
	for i in range(0, len(q_list)):
		p = []
		point = trajectory_msgs.msg.JointTrajectoryPoint()
		point.positions = list(q_list[i])
		for k in range(len(self.joint_names)):
	    		j = self.joint_names.index(self.current_joint_state.name[k])
	    		p.append(point.positions[j])
		point.positions = list(p)
		path.points.append(point)
	path.joint_names = self.current_joint_state.name
	return path		

    def mp_callback(self, data):
        T = tf.transformations.identity_matrix()
	q_s = []
        for i in range(0, len(self.joint_names)):
	    index = self.current_joint_state.name.index(self.joint_names[i])
	    q_s.append(self.current_joint_state.position[index])
        T_d = self.T_desire(data.translation.x, data.translation.y, data.translation.z, data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w)
	q_d = self.IK(T_d)
	if len(q_d)==0:
	    print "cannot solve IK to reach that point!!!"
	    return
	print "planning the path"
	q_list = self.motion_plannning(q_s, q_d)
	path = self.path(q_list)
	self.pub_trajectory.publish(path)

if __name__ == '__main__':
    rospy.init_node('mp', anonymous=True)
    ma = MoveArm()
    rospy.spin()
