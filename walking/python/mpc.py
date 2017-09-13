import numpy as np
np.set_printoptions(threshold=np.nan)

from scipy.linalg import block_diag
from qpoases import PyQProblem as QProblem
from qpoases import PyOptions as Options

class MPC(object):
   """MPC base class
   
   """

   def __init__(self, model, ntime, T, T_step, future_steps, feet):

      #linear system model
      self.model = model

      #feet dimensions
      self.feet = feet

      #walk parameters
      self.T = T
      self.T_step = T_step
      self.samples_per_step = int(self.T_step/self.T)
      self.future_steps = future_steps
      self.N = int(self.future_steps*self.samples_per_step)
      self.ntime = ntime

      #constraints (x, y)
      self.feet_ubounds = np.array([ 0.2,  -0.1569, 0.2, 0.25])
      self.feet_lbounds = np.array([-0.2,  -0.25,  -0.2, 0.1569])

      #compute the restrained CoP zone
      self.zone = (np.min(self.feet)/np.sqrt(2.))/2.
      self.CoP_ubounds = np.array([self.zone,  self.zone])
      self.CoP_lbounds = np.array([-self.zone, -self.zone])

      #angle between feet bound
      self.feet_angle_ubound =  0.05
      self.feet_angle_lbound = -0.05
      
      #number of decision variables (3*N for x, y, theta jerks + 3*future_steps
      #for x, y, theta for each of two steps)
      self.n_dec = 3*self.N + 3*self.future_steps 
      #number of constraints
      self.n_constraints = 2*self.N + 3*self.future_steps

      #QP
      self.QP = QProblem(self.n_dec, self.n_constraints)

      #initial state
      self.x = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.]])

      #initial CoP
      self.CoP = np.dot(self.model.D(self.T), self.x.T)

      #initial and future footstep position and orientation
      self.f_current = np.array([[0., 0., 0.]])
      #we record next step after each iteration and change it in the right moment
      self.f_future = np.array([[0., 0., 0.]])
      #controls
      self.controls = np.array([[0., 0., 0.]])
      #rotation matrix of the current footstep
      self.current_rotation = np.array([[ np.cos(self.f_current[0, 2]), np.sin(self.f_current[0, 2])],
                                       [ -np.sin(self.f_current[0, 2]), np.cos(self.f_current[0, 2])]])

      #initial step pattern - long vector of 1s and 0s which we roll over at each iteration
      self.cols_step0 = np.hstack((np.ones((1, self.samples_per_step)), np.zeros((1, self.samples_per_step))))
      self.cols_step1 = np.hstack((np.zeros((1, self.samples_per_step)), np.ones((1, self.samples_per_step))))
      self.cols_step2 = np.hstack((np.zeros((1, self.samples_per_step)), np.zeros((1, self.samples_per_step))))

      self.steps = np.hstack((self.cols_step0, self.cols_step1, self.cols_step2))

      #weights for QP
      self.alpha = 1e-5
      self.beta  = 1e3
      self.gamma = 2.5
      #old gains
      #self.alpha = 1e6
      #self.beta  = 1.0
      #self.gamma = 1e-6

   def condense(self):
      """obtain condensed system matrices
         as well as selection matrices

      """

      ############################iterate the linear model N times

      A = []
      B = []
      D = []

      for i in xrange(0, self.N):

         A.append(self.model.A(self.T))
         B.append(self.model.B(self.T))
         D.append(self.model.D(self.T))

      #################################################condense   

      #form Ux and Uu
      N = len(A)
      [Ns, Nu] = B[0].shape
      [Ds, Du] = D[0].shape

      #form Ux
      M = A[0]

      Ux = M.copy()

      for i in xrange(1, N):
         M = np.dot(A[i], M)
         Ux = np.vstack((Ux, M.copy()))

      #form Uu

      Uucol = np.vstack((np.zeros((Ns*(N-1), Ns)), np.eye(Ns)))
      Uu = np.zeros((Ns*N, Nu*N))
      Uu[:, Nu*(N - 1):Nu*N] = np.dot(Uucol, B[N - 1])

      for i in xrange(N, 0, -1):
         Uucol = np.dot(Uucol, A[i - 1])
         Uucol[Ns*(i - 1):Ns*i, :] = np.eye(Ns)
         Uu[:, Nu*(i - 1):Nu*i] = np.dot(Uucol, B[i - 1])

      #form D_diag

      D_diag = block_diag(*D)
      
      ##################################build MPC matrices

      #selecion matrix for x
      sx = [np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])]*self.N
      Sx = block_diag(*sx)

      #selection matrix for y
      sy = [np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0]])]*self.N
      Sy = block_diag(*sy)

      #selection matrix for xdot
      svx = [np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]])]*self.N
      Svx = block_diag(*svx)

      #selection matrix for ydot
      svy = [np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0]])]*self.N
      Svy = block_diag(*svy)

      #selection matrix for theta
      st = [np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0]])]*self.N
      St = block_diag(*st)

      #selection matrix for thetadot
      svt = [np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0]])]*self.N
      Svt = block_diag(*svt)

      #selection matrix for x CoP
      scx = [np.array([[1, 0]])]*self.N
      Scx = block_diag(*scx)

      #selection matrix for y CoP
      scy = [np.array([[0, 1]])]*self.N
      Scy = block_diag(*scy)

      return [Sx, Sy, Svx, Svy, St, Svt, Scx, Scy, Ux, Uu, D_diag]

   def rotate(self, point, theta):
      """apply the 2D rotation matrix to the point

      """

      point = point[:, np.newaxis]
      r_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
      p = np.dot(r_mat, point)

      return p

   def get_walk_matrices(self, i):
      """build the matrices V and Vdash

      """

      step0 = np.arange(0, self.N) 
      step1 = np.arange(self.N, 2*self.N) 
      step2 = np.arange(2*self.N, 3*self.N) 

      #handle the first step specially given that we change the foot after 7th iter
      if i == 0:
         self.steps = np.roll(self.steps, -1)

      #step change - go back to initial pattern
      if np.sum(self.steps[:, step0]) == 0.:
         self.steps = np.hstack((self.cols_step0, self.cols_step1, self.cols_step2))

      V = self.steps[:, step0].T
      Vdash = np.hstack((self.steps[:, step1].T, self.steps[:, step2].T)) 

      self.steps = np.roll(self.steps, -1)

      return [V, Vdash]

class RestrictedZoneMPC(MPC):
   """MPC with the restricted CoP zone on the foot
   
   """

   def solve(self, i, vref):
      """build the problem and solve it

      """

      #we build the N-sized reference vectors

      if(i + self.N < self.ntime + 1):
         #take the values stored in the rows (i to N)
         vref_pred = vref[i:i+self.N, :]
      else:
         # If the matrix doesn't have N rows ahead, then repeat the last row to fill the void
         vref_pred = np.vstack((vref[i:self.ntime, :], np.dot( np.ones((self.N - self.ntime + i, 1)),
                                                       np.expand_dims(vref[self.ntime - 1, :], axis=0))))

      #get condensed matrices (constant from on step to another)
      [Sx, Sy, Svx, Svy, St, Svt, Scx, Scy, Ux, Uu, D_diag] =  self.condense()

      #build matrix V
      V, V_dash = self.get_walk_matrices(i)

      ##################stuff for the solver###############################

      #build constraint vectors (changing from one step to another)
      
      #footsteps and CoP upper bounds

      #ubounds CoP x position
      ubounds1 = np.vstack(([self.CoP_ubounds[0] for _ in range(self.N)]))
      ubounds1 = ubounds1 - np.dot(np.dot(Scx, D_diag), np.dot(Ux, self.x.T)) + np.dot(V, self.f_current[0][0])

      #ubounds CoP y position
      ubounds2 = np.vstack([self.CoP_ubounds[1] for _ in range(self.N)])
      ubounds2 = ubounds2 - np.dot(np.dot(Scy, D_diag), np.dot(Ux, self.x.T)) + np.dot(V, self.f_current[0][1])

      #ubounds feet x position
      ubounds3 = np.vstack((self.feet_ubounds[0], self.feet_ubounds[2]))
      ubounds3 = ubounds3 + np.vstack((np.dot(self.current_rotation[np.newaxis, 0, :], self.f_current[0, 0:2, np.newaxis]), np.zeros((self.future_steps-1, 1))))

      #ubounds feet y position
      ubounds4 = np.vstack((self.feet_ubounds[1], self.feet_ubounds[3]))
      ubounds4 = ubounds4 + np.vstack((np.dot(self.current_rotation[np.newaxis, 1, :], self.f_current[0, 0:2, np.newaxis]), np.zeros((self.future_steps-1, 1))))
      
      #ubounds feet angles
      ubounds5 = self.feet_angle_ubound*np.ones((self.future_steps, 1))
      ubounds5 = ubounds5 + np.vstack((self.f_current[np.newaxis, np.newaxis, 0, 2], np.zeros((self.future_steps-1, 1))))

      ubA = np.vstack((ubounds1, ubounds2, ubounds3, ubounds4, ubounds5))

      #footsteps and CoP lower bounds

      #lbounds CoP x position
      lbounds1 = np.vstack([self.CoP_lbounds[0] for _ in range(self.N)])
      lbounds1 = lbounds1 - np.dot(np.dot(Scx, D_diag), np.dot(Ux, self.x.T)) + np.dot(V, self.f_current[0][0])

      #lbounds CoP y position
      lbounds2 = np.vstack([self.CoP_lbounds[1] for _ in range(self.N)])
      lbounds2 = lbounds2 - np.dot(np.dot(Scy, D_diag), np.dot(Ux, self.x.T)) + np.dot(V, self.f_current[0][1])      

      #lbounds feet x position
      lbounds3 = np.vstack((self.feet_lbounds[0], self.feet_lbounds[2]))
      lbounds3 = lbounds3 + np.vstack((np.dot(self.current_rotation[np.newaxis, 0, :], self.f_current[0, 0:2, np.newaxis]), np.zeros((self.future_steps-1, 1))))

      #lbounds feet y position
      lbounds4 = np.vstack((self.feet_lbounds[1], self.feet_lbounds[3]))
      lbounds4 = lbounds4 + np.vstack((np.dot(self.current_rotation[np.newaxis, 1, :], self.f_current[0, 0:2, np.newaxis]), np.zeros((self.future_steps-1, 1))))

      #lbounds feet angles
      lbounds5 = self.feet_angle_lbound*np.ones((self.future_steps, 1))
      lbounds5 = lbounds5 + np.vstack((self.f_current[np.newaxis, np.newaxis, 0, 2], np.zeros((self.future_steps-1, 1))))

      lbA = np.vstack((lbounds1, lbounds2, lbounds3, lbounds4, lbounds5))

      #big A matrix (constraints matrix)
      A = np.zeros((self.n_constraints, self.n_dec))

      #A elements for constraints for CoP position
      A[0:self.N, :] = np.hstack((np.dot(np.dot(Scx, D_diag), Uu), -V_dash, np.zeros((self.N, 2*self.future_steps))))
      A[self.N:2*self.N, :] = np.hstack((np.dot(np.dot(Scy, D_diag), Uu), np.zeros((self.N, self.future_steps)), -V_dash, np.zeros((self.N, self.future_steps))))

      A[2*self.N, 3*self.N] = self.current_rotation[0, 0]
      A[2*self.N, 3*self.N+2] = self.current_rotation[0, 1]

      A[2*self.N+1, 3*self.N] = -self.current_rotation[0, 0]
      A[2*self.N+1, 3*self.N+1] = self.current_rotation[0, 0]
      A[2*self.N+1, 3*self.N+2] = -self.current_rotation[0, 1]
      A[2*self.N+1, 3*self.N+3] = self.current_rotation[0, 1]

      A[2*self.N+2, 3*self.N] = self.current_rotation[1, 0]
      A[2*self.N+2, 3*self.N+2] = self.current_rotation[1, 1]

      A[2*self.N+3, 3*self.N] = -self.current_rotation[1, 0]
      A[2*self.N+3, 3*self.N+1] = self.current_rotation[1, 0]
      A[2*self.N+3, 3*self.N+2] = -self.current_rotation[1, 1]
      A[2*self.N+3, 3*self.N+3] = self.current_rotation[1, 1]

      #max angle between feet constraints
      A[2*self.N+4, 3*self.N+4] = 1
      A[2*self.N+5, 3*self.N+4] = -1
      A[2*self.N+5, 3*self.N+5] = 1

      #QP solver takes one dim arrays
      lbA = lbA.reshape((lbA.shape[0],))
      ubA = ubA.reshape((ubA.shape[0],))

      #############################################################################HESSIAN AND GRADIENT####################

      H = np.zeros((self.n_dec, self.n_dec))
      H[0:3*self.N, 0:3*self.N] = self.alpha*np.ones((3*self.N, 3*self.N)) + self.beta*np.dot(np.dot(Svx, Uu).T, np.dot(Svx, Uu)) + self.beta*np.dot(np.dot(Svy, Uu).T, np.dot(Svy, Uu)) + \
                                      + self.beta*np.dot(np.dot(Svt, Uu).T, np.dot(Svt, Uu)) + self.gamma*np.dot(np.dot(np.dot(Scx, D_diag), Uu).T, np.dot(np.dot(Scx, D_diag), Uu)) + \
                                      + self.gamma*np.dot(np.dot(np.dot(Scy, D_diag), Uu).T, np.dot(np.dot(Scy, D_diag), Uu)) + self.gamma*np.dot(np.dot(St, Uu).T, np.dot(St, Uu))

      H[0:3*self.N, 3*self.N:] = np.hstack((-self.gamma*np.dot(np.dot(np.dot(Scx, D_diag), Uu).T, V_dash), -self.gamma*np.dot(np.dot(np.dot(Scy, D_diag), Uu).T, V_dash), -self.gamma*np.dot(np.dot(St, Uu).T, V_dash)))
      H[3*self.N:, 0:3*self.N] = np.vstack((-self.gamma*np.dot(V_dash.T, np.dot(np.dot(Scx, D_diag), Uu)), -self.gamma*np.dot(V_dash.T, np.dot(np.dot(Scy, D_diag), Uu)), -self.gamma*np.dot(V_dash.T, np.dot(St, Uu))))
      H[3*self.N:, 3*self.N:] = self.gamma*block_diag(np.dot(V_dash.T, V_dash), np.dot(V_dash.T, V_dash), np.dot(V_dash.T, V_dash))

      g = np.zeros((self.n_dec, 1))
      g[0:3*self.N, :] = self.beta*np.dot(np.dot(Svx, Uu).T, np.dot(np.dot(Svx, Ux), self.x.T) - vref_pred[:, 0][np.newaxis].T) + self.beta*np.dot(np.dot(Svy, Uu).T, np.dot(np.dot(Svy, Ux), self.x.T) - vref_pred[:, 1][np.newaxis].T) + \
                             self.beta*np.dot(np.dot(Svt, Uu).T, np.dot(np.dot(Svt, Ux), self.x.T) - vref_pred[:, 2][np.newaxis].T) + self.gamma*np.dot(np.dot(np.dot(Scx, D_diag), Uu).T, np.dot(np.dot(np.dot(Scx, D_diag), Ux), self.x.T) - np.dot(V, self.f_current[0][0])) + \
                             + self.gamma*np.dot(np.dot(np.dot(Scy, D_diag), Uu).T, np.dot(np.dot(np.dot(Scy, D_diag), Ux), self.x.T) - np.dot(V, self.f_current[0][1])) + self.gamma*np.dot(np.dot(St, Uu).T, np.dot(np.dot(St, Ux), self.x.T)) - \
                             - self.gamma*np.dot(np.dot(St, Uu).T, np.dot(V, self.f_current[0][2]))

      g[3*self.N:50, :] = -self.gamma*np.dot(V_dash.T, np.dot(np.dot(np.dot(Scx, D_diag), Ux), self.x.T) - np.dot(V, self.f_current[0][0]))
      g[50:52, :] = -self.gamma*np.dot(V_dash.T, np.dot(np.dot(np.dot(Scy, D_diag), Ux), self.x.T) - np.dot(V, self.f_current[0][1]))
      g[52:, :] = -self.gamma*np.dot(V_dash.T, np.dot(np.dot(St, Ux), self.x.T) - np.dot(V, self.f_current[0][2]))
      
      g = g.reshape((g.shape[0],))
      
      ###########################################################################################################################

      #solver options - MPC
      myOptions = Options()
      myOptions.setToMPC()
      self.QP.setOptions(myOptions)

      #setting lb and ub to huge numbers - otherwise solver gives errors
      lb = -1000000000.*np.ones((self.n_dec,))
      ub =  1000000000.*np.ones((self.n_dec,))

      #solve QP - QP.init()
      self.QP.init(H, g, A, lb, ub, lbA, ubA, nWSR=np.array([1000000]))

      #get the solution (getPrimalSolution())
      x_opt = np.zeros(self.n_dec)
      self.QP.getPrimalSolution(x_opt)

      #update the state of the system, CoP and the footstep (take the first elements of the solution)
      self.controls = np.array([[x_opt[0], x_opt[1], x_opt[2]]])

      self.x = (np.dot(self.model.A(self.T), self.x.T) + np.dot(self.model.B(self.T), self.controls.T)).T
      self.CoP = np.dot(self.model.D(self.T), self.x.T)

      #first future step
      self.f_future = np.array([[x_opt[3*self.N], x_opt[3*self.N+self.future_steps], x_opt[3*self.N+2*self.future_steps]]])

      #change the current (every 8 samples except the first step) foot position
      if np.mod(i, self.samples_per_step) == self.samples_per_step-2:
         self.f_current = self.f_future
         #rotate the footstep bounds
         self.current_rotation = np.array([[ np.cos(self.f_current[0, 2]), np.sin(self.f_current[0, 2])],
                                          [ -np.sin(self.f_current[0, 2]), np.cos(self.f_current[0, 2])]])
         #swap footstep constraints
         tmp1, tmp2 = self.feet_ubounds[1], self.feet_lbounds[1]  
         self.feet_ubounds[1], self.feet_lbounds[1] = self.feet_ubounds[3], self.feet_lbounds[3]
         self.feet_ubounds[3], self.feet_lbounds[3] = tmp1, tmp2

      #return all the variables of interest
      return [self.x, self.f_current, self.CoP, self.controls]

class LinearizedMPC(MPC):
   """MPC with linearized CoP and footstep constraints 

   """
   
   #implement solve function with linearization here

   pass

