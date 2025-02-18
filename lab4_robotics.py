from lab2_robotics import * # Includes numpy import

def jacobianLink(T, revolute, link): # Needed in Exercise 2
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        link(integer): index of the link for which the Jacobian is computed

        Returns:
        (Numpy array): end-effector Jacobian
    '''

    # 1. Initialize J and O.
    J = np.zeros((6, len(revolute)))  # -1 because T includes the base transformation, which adds 1 more matrix
    O = T[-1][:3, -1].reshape((3, 1))  # from base to n

    # 2. For each joint of the robot
    for i in range(len(revolute)):
        if i < link:  #(For joints before the specified link compute jacobian because they contribute to end-effector motion)
            # a. Extract z and o.
            Ti = T[i]
            z = Ti[:3, 2].reshape((3, 1))
            o = Ti[:3, -1].reshape((3, 1))  # from base to i-1

            # b. Check joint type.
            rhoi = int(revolute[i])

            # c. Modify corresponding column of J.
            Ji = np.block([
                [np.array([[np.cross(rhoi * z, (O - o), axis=0) + (1 - rhoi) * z]])],
                [np.array([[rhoi * z]])]
            ])
            J[:, i] = Ji.reshape((6,))

        else: # Joints beyond the specific linnk do not contribute to robot's motion hence corresponding jacobian columns are 0
            J[:, i] = np.zeros(6)

    return J


'''
    Class representing a robotic manipulator.
'''
class Manipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)

    '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    '''
    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    '''
        Method that returns the end-effector transformation.
    '''
    def getEETransform(self):
        return self.T[-1]

    '''
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    '''
    def getJointPos(self, joint):
        return self.q[joint]

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof
    
    '''
        Method to get the transformation for a selected link.
    '''
    def getLinkTransform(self, link):
        return self.T[link]

    '''
       Method to get the Jacobian for a selected link.
    '''
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revolute, link)


'''
    Base class representing an abstract Task.
'''
class Task:
    '''
        Constructor.

        Arguments:
        name (string): title of the task
        desired (Numpy array): desired sigma (goal)
    '''
    def __init__(self, name, desired):
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        self.feedforward_velocity = np.zeros((6,1))
        self.K = np.eye(self.sigma_d.shape[0])
        self.active = True

        
    '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    '''
    def update(self, robot):
        pass

    ''' 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
    '''
    def setDesired(self, value):
        self.sigma_d = value

    '''
        Method returning the desired sigma.
    '''
    def getDesired(self):
        return self.sigma_d

    '''
        Method returning the task Jacobian.
    '''
    def getJacobian(self):
        return self.J

    '''
        Method returning the task error (tilde sigma).
    '''    
    def getError(self):
        return self.err
    
    '''
        Method to set the feedforward velocity vector.
    '''
    def setFeedforwardVelocity(self, velocity):
        self.feedforward_velocity = velocity

    '''
        Method to get the feedforward velocity vector.
    '''
    def getFeedforwardVelocity(self):
        return self.feedforward_velocity

    '''
        Method to set gain matrix K.
    '''
    def setGainMatrix(self, K):
        self.K = K

    '''
        Method to get gain matrix K.
    '''
    def getGainMatrix(self):
        return self.K
    
    '''
        Method to check if the task is active.
    '''
    def isActive(self, robot):
        return self.active


'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, desired, link=3):
        super().__init__(name, desired)
        self.link = link
        self.J = np.zeros((2, self.link))
        self.feedforward_velocity = np.zeros((2, 1))  
        self.K = np.eye(2)

    def update(self, robot):
        J_link = robot.getLinkJacobian(self.link)  
        self.J = J_link[:2, :]  
        link_position = robot.getLinkTransform(self.link)[:2, -1].reshape((2, 1))  
        self.err = self.getDesired() - link_position  


'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired, link=3):
        super().__init__(name, desired)
        self.link = link
        self.J = np.zeros((1, self.link))  
        self.feedforward_velocity = np.zeros((1, 1)) 
        self.K = np.eye(1)  

    def update(self, robot):
        J_link = robot.getLinkJacobian(self.link)
        self.J = np.array([J_link[-1, :]])  # last row of of the Jacobian, corresponds to rotation around z
        link_orientation = robot.getLinkTransform(self.link)[:2, :2]
        self.err = self.getDesired() - np.arctan2(link_orientation[1,0], link_orientation[0,0])


'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired, link=3):
        super().__init__(name, desired)
        self.link = link
        self.J = np.zeros((3, self.link + 1)) 
        self.feedforward_velocity = np.zeros((3, 1))  
        self.K = np.eye(3)  

    def update(self, robot):
        J_link = robot.getLinkJacobian(self.link)
        J_pos = J_link[:2, :]
        J_ori = np.array(J_link[-1,:])
        self.J = np.vstack((J_pos, J_ori))

        # calculate task error
        link_transform = robot.getLinkTransform(self.link)
        link_position = link_transform[:2, -1].reshape((2, 1)) 
        link_orientation = np.arctan2(link_transform[1, 0], link_transform[0, 0])
        self.err = np.vstack((self.getDesired()[:2] - link_position, self.getDesired()[2] - link_orientation))

''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, desired):
        super().__init__(name, desired)
        self.J = np.zeros((1,3)) # Initialize with proper dimensions
        self.err = np.zeros((1,1)) # Initialize with proper dimensions
        self.feedforward_velocity = np.zeros_like(desired)
        self.K = np.eye(len(desired))
        
    def update(self, robot):
        self.J = np.array([[1,0,0]]) # Update task Jacobian
        self.err = self.getDesired() - robot.q[0] # Update task error

class Obstacle2D(Task):
    def __init__(self, name, obstacle_pos, limits):
        super().__init__(name, obstacle_pos) # self.sigma_d becomes the obstacle position
        self.J = np.zeros((2,3))
        self.err = np.zeros((2,1)) # distance between the obstacle and EE
        self.ra = limits[0] # activation threshold
        self.rd = limits[1] # deactivation threshold

        self.feedforward_velocity = np.zeros((2,1)) # not really relevant but added to maintain generality
        self.K = np.eye(2)

        self.active = False # set initial activation to False

    def update(self, robot):
        self.J = robot.getEEJacobian()[:2] # the first 2 rows of the EE Jacobian
        EEpos = robot.getEETransform()[:2,-1].reshape((2,1))
        self.err = (EEpos - self.getDesired())/np.linalg.norm(EEpos - self.getDesired())

    def isActive(self, robot): #override the base class
        EEpos = robot.getEETransform()[:2,-1].reshape((2,1))
        if (self.active == False) and (np.linalg.norm(EEpos - self.getDesired()) <= self.ra):
            self.active = True
        elif (self.active == True) and (np.linalg.norm(EEpos - self.getDesired()) >= self.rd):
            self.active = False
        
        return self.active
    
class JointLimits(Task):
    def __init__(self, name, limits):
        super().__init__(name, np.zeros((0,1))) # no sigma_d
        self.J = np.zeros((1,3))
        self.err = 1 # xdot
        self.qmin = limits[0] # lower limit
        self.qmax = limits[1] # upper limit
        self.ra = 0.02 # activation threshold
        self.rd = 0.04 # deactivation threshold

        self.feedforward_velocity = 0 # not really relevant but added to maintain generality
        self.K = np.eye(1)

        self.active = 0 # initially set to False (inactive)

    def update(self, robot):
        self.J = np.array([[1,0,0]])
        self.err = np.array([[self.active]])

    def isActive(self, robot): #override the base class
        qi = robot.q[0]
        if (self.active == 0):
            if (qi >= self.qmax-self.ra):
                self.active = -1
            elif (self.active == 0) and (qi <= self.qmin+self.ra):
                self.active = 1
        elif (self.active == -1) and (qi <= self.qmax-self.rd):
            self.active = 0
        elif (self.active == 1) and (qi >= self.qmin+self.rd):
            self.active = 0
            
        return self.active


def WeightedDLS(A, damping, W):
    '''
        Function computes the weighted damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor
        W (Numpy array): weighting matrix

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    return np.linalg.inv(W) @ A.T @ np.linalg.inv(A@np.linalg.inv(W)@A.T + damping**2*np.eye(A.shape[0]))
