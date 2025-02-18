from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patch

# Robot model
d = np.zeros(3)                  # displacement along Z-axis
theta = np.array([0.2,0.5,0.1])  # rotation around Z-axis
alpha = np.zeros(3)              # rotation around X-axis
a = np.array([0.75,0.5,0.3])     # displacement along X-axis
revolute = [True,True,True]      # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object

# Task hierarchy definition
obstacle1_pos = np.array([0.0, 1.0]).reshape(2,1)
obstacle1_r = 0.5

obstacle2_pos = np.array([0.8,-0.5]).reshape(2,1)
obstacle2_r = 0.3

obstacle3_pos = np.array([-0.5,-0.7]).reshape(2,1)
obstacle3_r = 0.4

obstacle123_r = [obstacle1_r,obstacle2_r,obstacle3_r] # for plotting purposes

tasks = [ 
          Obstacle2D("Obstacle 1 avoidance", obstacle1_pos, np.array([obstacle1_r, obstacle1_r+0.02])),
          Obstacle2D("Obstacle 2 avoidance", obstacle2_pos, np.array([obstacle2_r, obstacle2_r+0.02])),
          Obstacle2D("Obstacle 3 avoidance", obstacle3_pos, np.array([obstacle3_r, obstacle3_r+0.02])),
          Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2,1))
        ] 

# Simulation params
dt = 1.0/60.0

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.add_patch(patch.Circle(obstacle1_pos.flatten(), obstacle1_r, color='red', alpha=0.3))
ax.add_patch(patch.Circle(obstacle2_pos.flatten(), obstacle2_r, color='purple', alpha=0.3))
ax.add_patch(patch.Circle(obstacle3_pos.flatten(), obstacle3_r, color='green', alpha=0.3))
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target

PPx = []
PPy = []

# initialize a memory to store the error evolution
error_evol = []
if isinstance(tasks[0],Configuration2D):
    error_evol = [[],[]] # initialize 2 lists to store position and orientation
else:
    for i in range(len(tasks)):
        error_evol.append([])

# Weighting matrix
W = np.diag([5,3,1])

# Simulation initialization
def init():
    global tasks
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])

    # initialize random desired EE position and orientation
    tasks[3].setDesired(np.random.uniform(-1.2, 1.2, (2, 1)))

    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    global error_evol
    
    ### Recursive Task-Priority algorithm (w/set-based tasks)
    # The algorithm works in the same way as in Lab4. 
    # The only difference is that it checks if a task is active.
    ###
    # Initialize null-space projector
    Pi_1 = np.eye(robot.getDOF())
    
    # Initialize output vector (joint velocity)
    dqi_1 = np.zeros((robot.getDOF(), 1))
    
    # Loop over tasks
    for i in range(len(tasks)):
        task = tasks[i]

        # Update task state
        task.update(robot)
        Ji = task.getJacobian()
        erri = task.getError()

        # store the error evolution
        if isinstance(task,Obstacle2D):
            EEpos = robot.getEETransform()[:2,-1].reshape((2,1))
            error_evol[i].append(np.linalg.norm(EEpos - task.getDesired())- obstacle123_r[i])
        elif erri.shape[0] > 1:
            error_evol[i].append(np.linalg.norm(erri))
        else:
            error_evol[i].append(erri[0,0])

        # Skip if the task is not active
        if not task.isActive(robot):
            continue

        # Get feedforward velocity and gain matrix
        feedforward_velocity = task.getFeedforwardVelocity()
        K = task.getGainMatrix()

        xdoti = feedforward_velocity + K @ erri

        # Compute augmented Jacobian
        Ji_bar = Ji @ Pi_1
        # Compute task velocity with feedforward term and K matrix
        # dqi = WeightedDLS(Ji_bar, 0.1, W) @ (xdoti - Ji @ dqi_1)
        dqi = DLS(Ji_bar, 0.1) @ (xdoti - Ji @ dqi_1)
        dq = dqi_1 + dqi
        # Update null-space projector
        Pi = Pi_1 - np.linalg.pinv(Ji_bar) @ Ji_bar

        # Store the current P and dq for the next iteration
        Pi_1 = Pi
        dqi_1 = dqi


    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[3].getDesired()[0], tasks[3].getDesired()[1])
    
    return line, path, point

# Function to generate the second plot
def error_plot(tasks, error_evol):
    # define simulation time (after repeats)
    tfinal = len(error_evol[0])*dt
    tt = np.arange(0,tfinal,dt)

    # Create a second plot for the joint positions
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, xlim=(0, tfinal))
    ax2.set_title('Task Priority - Error Evolution')
    ax2.set_xlabel('Time[s]')
    ax2.set_ylabel('Error')
    # ax2.set_aspect('equal')
    ax2.grid()

    labels = ["d_1 (distance to obstacle 1)",
              "d_2 (distance to obstacle 2)",
              "d_3 (distance to obstacle 3)",
              "e_1 (end-effector position error)"] # list of labels for legend
    
    # color settings
    colors = ['red','purple','green','blue']

    for i in range(len(error_evol)):
        ax2.plot(tt, error_evol[i], color=colors[i])

    ax2.legend(labels)
    plt.show()


# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Show the plot of error evolution
error_plot(tasks,error_evol)