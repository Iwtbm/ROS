#!/usr/bin/env python  
import rospy
import math
import numpy
from urdf_parser_py.urdf import URDF
from geometry_msgs.msg import Transform
from cartesian_control.msg import CartesianCommand
from sensor_msgs.msg import JointState
import tf
import tf2_ros
import geometry_msgs.msg
robot = URDF.from_parameter_server()
joint_transforms = []
x_current = []
j_t = []
T_d = []
T_xd = []
current_joint_state = []
q = 0
secondary = False
joint_names = []

def rotation_from_matrix(matrix):
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

# Get position(x, y ,z, rx, ry, rz) from transform matrix
def position_from_matrix(matrix):
    angle, axis = rotation_from_matrix(matrix)
    rotation_change = numpy.dot(angle, axis)
    rotation_change = rotation_change.reshape(3,1)
    translation_change = tf.transformations.translation_from_matrix(matrix)
    translation_change = translation_change.reshape(3,1)
    x = numpy.row_stack((translation_change, rotation_change))
    return x
# Get information from URDF
def get_joint_info():
	global joint_names
        num_joints = 0
        joint_names = []
        joint_axes = []
        link = robot.get_root()
        while True:
            if link not in robot.child_map: break            
            (joint_name, next_link) = robot.child_map[link][0]
            current_joint = robot.joint_map[joint_name]        
            if current_joint.type != 'fixed':
                num_joints = num_joints + 1
                joint_names.append(current_joint.name)
                joint_axes.append(current_joint.axis)
            link = next_link
	return (joint_names, joint_axes)

# cartesian_control method
def cartesian_control(T_c, T_d, joint_transforms, q0_desire, secondary):
    v = []
    V_j = []
    qd = []
    iT_c = tf.transformations.inverse_matrix(T_c)
    delta_x = numpy.dot(iT_c, T_d)
    x_dot = position_from_matrix(delta_x)
    p = [x_dot[0], x_dot[1], x_dot[2]]
    k = [x_dot[3], x_dot[4], x_dot[5]]
    if numpy.linalg.norm(p) > 0.1:                             #scale velosity
        p /= 10 * max(numpy.abs(p))
    if numpy.linalg.norm(k) > 1.0:                             #scale velosity
        k /= max(numpy.abs(k))
    x_dot = numpy.row_stack((p, k))
    v_ee = x_dot.reshape(6, 1)              
    for jt in joint_transforms:
	jTee = numpy.dot(tf.transformations.inverse_matrix(jt), T_c)
    	eeTj = numpy.dot(iT_c, jt)
	Sj = [[0, -jTee[2, 3], jTee[1, 3]], [jTee[2, 3], 0, -jTee[0, 3]], [-jTee[1, 3], jTee[0, 3], 0]]
    	V_j.append(numpy.row_stack((numpy.column_stack((eeTj[:3, :3], -numpy.dot(eeTj[:3, :3], Sj))), numpy.column_stack((numpy.zeros((3,3)), eeTj[:3, :3])))))
    (joint_names, joint_axes) =get_joint_info()
    for i in range(0, len(joint_axes)):
	for j in range(0, len(joint_axes[i])):
	    if joint_axes[i][j] != 0:
		v.append(V_j[i][:, j+3]*joint_axes[i][j])	            
    J = v[0].reshape(6, 1)
    for i in range(1, len(v)):
	J = numpy.column_stack((J, v[i].reshape(6, 1)))
    Jps = numpy.linalg.pinv(J, 0.01)
    q_des = numpy.dot(Jps, v_ee) 
    for i in range(0, len(q_des)):
 	qd.append(q_des[i])

    if secondary == True:
	q_sec = numpy.zeros((len(joint_names),1))
	q_sec[0] = q0_desire - current_joint_state.position[0]
	q_null = numpy.dot((numpy.identity(len(joint_names)) - numpy.dot(numpy.linalg.pinv(J), J)), q_sec)
	qd = qd + q_null				
    if numpy.linalg.norm(qd) > 1.0:                             #scale velosity
        qd /= max(numpy.abs(qd))
    return qd

# Find Jacobian pseudoinverse for inverse_kinematics
def inverse_kinematics(T_c, T_d, joint_transforms):
    v = []
    V_j = []
    iT_c = tf.transformations.inverse_matrix(T_c)
    delta_x = numpy.dot(iT_c, T_d)
    x_dot = position_from_matrix(delta_x)
    for jt in joint_transforms:
	jTee = numpy.dot(tf.transformations.inverse_matrix(jt), T_c)
    	eeTj = numpy.dot(iT_c, jt)
	Sj = [[0, -jTee[2, 3], jTee[1, 3]], [jTee[2, 3], 0, -jTee[0, 3]], [-jTee[1, 3], jTee[0, 3], 0]]
    	V_j.append(numpy.row_stack((numpy.column_stack((numpy.zeros((3,3)), -numpy.dot(eeTj[:3, :3], Sj))), numpy.column_stack((numpy.zeros((3,3)), eeTj[:3, :3])))))
    (joint_names, joint_axes) =get_joint_info()
    for i in range(0, len(joint_axes)):
	for j in range(0, len(joint_axes[i])):
	    if joint_axes[i][j] != 0:
		v.append(V_j[i][:, j+3]*joint_axes[i][j])	            
    J = v[0].reshape(6, 1)
    for i in range(1, len(v)):
	J = numpy.column_stack((J, v[i].reshape(6, 1)))
    Jp = numpy.linalg.pinv(J)
    return (Jp, x_dot)

# Find every joint transforms (from base to joint) and current end effector transform, 
# this is similar with professor writing in 'marker_control', but is not the same.
def process_link_recursive(link, T, joint_values, qc, judge):
	global x_current
        if link not in robot.child_map: 
            x_current = T
	    return   
        for i in range(0,len(robot.child_map[link])):
            (joint_name, next_link) = robot.child_map[link][i]
            if joint_name not in robot.joint_map:
                rospy.logerror("Joint not found in map")
                continue
            current_joint = robot.joint_map[joint_name]        

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
                joint_transforms.append(current_joint_T)
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

            process_link_recursive(next_link, next_link_T, joint_values, qc, judge)
# Transfer desired position to matrix
def T_desire(x, y, z, rx, ry, rz, w):
    T_d = tf.transformations.concatenate_matrices(
	tf.transformations.translation_matrix((x, y, z)),
	tf.transformations.quaternion_matrix((rx, ry, rz, w))
        )
    return(T_d)

# cartesian_command call back
def cc_callback(data):
    global T_d, current_joint_state, joint_transforms, x_current, q, secondary
    T_d = T_desire(data.x_target.translation.x, data.x_target.translation.y, data.x_target.translation.z, data.x_target.rotation.x, data.x_target.rotation.y, data.x_target.rotation.z, 
        data.x_target.rotation.w)
    T = tf.transformations.identity_matrix()
    joint_transforms = []
    process_link_recursive(robot.get_root(), T, current_joint_state, qc = 1, judge = True)
    secondary = False
    if data.secondary_objective == True:
	q = data.q0_target
	secondary = True
    talker()

# joint_states call back
def js_callback(joint_values):
    global current_joint_state  
    current_joint_state = joint_values

# ik_command call back
def ik_callback(transform):
    global x_current, joint_transforms, current_joint_state
    T_xd = T_desire(transform.translation.x, transform.translation.y, transform.translation.z, transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w)
    q = []
    (joint_names, joint_axes) = get_joint_info()
    n = len(joint_names)
    qc = numpy.zeros((n, 1))
    for i in range(0, n):
	qc[i] = 2 * numpy.random.uniform(0, numpy.pi)
    delta_q = numpy.ones((n, 1))
    dx = 1
    j = 0
    k = 0
    while dx > 0.01:
        absdx = []
	absdth = []
	T = tf.transformations.identity_matrix()
    	joint_transforms = []
    	process_link_recursive(robot.get_root(), T, current_joint_state, qc, judge = False)
	print(robot.get_root())
    	(Jp, x_dot) = inverse_kinematics(x_current, T_xd, joint_transforms)
    	delta_q = numpy.dot(Jp, x_dot)
        dx = numpy.amax(numpy.abs(x_dot))
	for i in range(0, n):
	    index = current_joint_state.name.index(joint_names[i])
	    qc[index] = qc[index] + delta_q[i]
        j = j + 1
	if j > 300:
	    for i in range(0, n):
		qc[i] = 2 * numpy.random.uniform(0, numpy.pi)
	    j = 0
            k = k+1
        if k > 20:
            raise ValueError('Cannot reach that point!')
    for i in range(0, n):
	index = current_joint_state.name.index(joint_names[i])
        q.append(qc[index])
    pub = rospy.Publisher("/joint_command", JointState, queue_size=1)
    msg = JointState()
    msg.position = q
    msg.name = joint_names
    pub.publish(msg)

def talker():
    global x_current, T_d, q, joint_transforms, secondary, joint_names
    pub = rospy.Publisher("/joint_velocities", JointState, queue_size=1)
    msg = JointState()
    msg.velocity = cartesian_control(x_current, T_d, joint_transforms, q, secondary)
    msg.name = joint_names
    pub.publish(msg)

def listener():
    rospy.Subscriber("/joint_states", JointState, js_callback)
    rospy.Subscriber('/cartesian_command', CartesianCommand, cc_callback)
    rospy.Subscriber('/ik_command', Transform, ik_callback)

if __name__ == '__main__':
    rospy.init_node('ccik')
    listener()
    rospy.spin()
