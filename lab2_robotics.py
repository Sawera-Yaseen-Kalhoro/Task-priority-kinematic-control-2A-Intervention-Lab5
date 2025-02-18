import numpy as np # Import Numpy
from numpy import cos,sin

def DH(d, theta, a, alpha):
    '''
        Function builds elementary Denavit-Hartenberg transformation matrices 
        and returns the transformation matrix resulting from their multiplication.

        Arguments:
        d (double): displacement along Z-axis
        theta (double): rotation around Z-axis
        a (double): displacement along X-axis
        alpha (double): rotation around X-axis

        Returns:
        (Numpy array): composition of elementary DH transformations
    '''
    # 1. Build matrices representing elementary transformations (based on input parameters).
    Td = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,d],[0,0,0,1]])
    Ttheta = np.array([[cos(theta),-sin(theta),0,0],[sin(theta),cos(theta),0,0],[0,0,1,0],[0,0,0,1]])
    Ta = np.array([[1,0,0,a],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    Talpha = np.array([[1,0,0,0],[0,cos(alpha),-sin(alpha),0],[0,sin(alpha),cos(alpha),0],[0,0,0,1]])

    # 2. Multiply matrices in the correct order (result in T).
    T = Td @ Ttheta @ Ta @ Talpha
    return T

def kinematics(d, theta, a, alpha):
    '''
        Functions builds a list of transformation matrices, for a kinematic chain,
        descried by a given set of Denavit-Hartenberg parameters. 
        All transformations are computed from the base frame.

        Arguments:
        d (list of double): list of displacements along Z-axis
        theta (list of double): list of rotations around Z-axis
        a (list of double): list of displacements along X-axis
        alpha (list of double): list of rotations around X-axis

        Returns:
        (list of Numpy array): list of transformations along the kinematic chain (from the base frame)
    '''
    T = [np.eye(4)] # Base transformation
    Toi = T[0] # initialize memory variable

    # For each set of DH parameters:
    for (di,thetai,ai,alphai) in zip(d,theta,a,alpha):
        # 1. Compute the DH transformation matrix.
        Ti = DH(di,thetai,ai,alphai)

        # 2. Compute the resulting accumulated transformation from the base frame.
        Toi = Toi @ Ti
        
        # 3. Append the computed transformation to T.
        T.append(Toi)

    return T

# Inverse kinematics
def jacobian(T, revolute):
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # 1. Initialize J and O.
    J = np.zeros((6,len(T)-1)) #-1 because T includes the base transformation, which adds 1 more matrix
    O = T[-1][:3,-1].reshape((3,1)) # from base to n

    # 2. For each joint of the robot
    for i in range(len(T)-1):
        # a. Extract z and o.
        Ti = T[i]
        z = Ti[:3,2].reshape((3,1))
        o = Ti[:3,-1].reshape((3,1)) # from base to i-1

        # b. Check joint type.
        rhoi = int(revolute[i])

        # c. Modify corresponding column of J.
        Ji = np.block([[np.array([[np.cross(rhoi*z,(O-o),axis=0) + (1-rhoi)*z]])],[np.array([[rhoi*z]])]])
        J[:,i] = Ji.reshape((6,))
    return J

# Damped Least-Squares
def DLS(A, damping):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    return A.T @ np.linalg.inv(A@A.T + damping**2*np.eye(A.shape[0])) # Implement the formula to compute the DLS of matrix A.


# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    '''
        Function extracts the characteristic points of a kinematic chain on a 2D plane,
        based on the list of transformations that describe it.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    
        Returns:
        (Numpy array): an array of 2D points
    '''
    P = np.zeros((2,len(T)))
    for i in range(len(T)):
        P[:,i] = T[i][0:2,3]
    return P