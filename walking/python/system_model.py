import numpy as np

#model of the linear system

class SystemModel(object):

    #gravity

    _g = 9.81

    def __init__(self, h_CoM):

        self.h_CoM = h_CoM

    def A(self, T):

        M = np.array([[1,  T,  T**2/2],
                      [0,  1,       T],
                      [0,  0,       1]])

        A_row1 = np.hstack((M, np.zeros((3, 3)), np.zeros((3, 3))))  
        A_row2 = np.hstack((np.zeros((3, 3)), M, np.zeros((3, 3))))  
        A_row3 = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), M))  

        A = np.vstack((A_row1, A_row2, A_row3))

        return A

    def B(self, T):

        M_col1 = np.array([[T**3/6, T**2/2, T, 0, 0, 0, 0, 0, 0]]).T
        M_col2 = np.array([[0, 0, 0, T**3/6, T**2/2, T, 0, 0, 0]]).T
        M_col3 = np.array([[0, 0, 0, 0, 0, 0, T**3/6, T**2/2, T]]).T

        B = np.hstack((M_col1, M_col2, M_col3))

        return B

    def D(self, T):

        D_row1 = np.array([[1, 0, -self.h_CoM/SystemModel._g, 0, 0, 0, 0, 0, 0]])
        D_row2 = np.array([[0, 0, 0, 1, 0, -self.h_CoM/SystemModel._g, 0, 0, 0]])

        D = np.vstack((D_row1, D_row2))

        return D

class SystemModelDCM(object):

    #gravity

    _g = 9.81

    def __init__(self, h_CoM):

        self.h_CoM = h_CoM
        self.omega = np.sqrt(SystemModelDCM._g / h_CoM) 

    def A(self, T):

        M = np.array([[1,  T,  (1.0 / (self.omega**2)) * (np.cosh(self.omega * T) - 1.0)],
                      [0,  1,  (1.0 / self.omega) * np.sinh(self.omega * T)             ],
                      [0,  0,  np.cosh(self.omega * T)                                  ]])

        A_row1 = np.hstack((M, np.zeros((3, 3)), np.zeros((3, 3))))  
        A_row2 = np.hstack((np.zeros((3, 3)), M, np.zeros((3, 3))))  
        A_row3 = np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), M))  

        A = np.vstack((A_row1, A_row2, A_row3))

        return A

    def B(self, T):

        M_col1 = np.array([[(1.0 / (self.omega**3))*(np.sinh(self.omega*T) - T), (1.0 / (self.omega**2))*(np.cosh(self.omega*T) - 1.0), (1.0 / self.omega)*np.sinh(self.omega*T), 0, 0, 0, 0, 0, 0]]).T
        M_col2 = np.array([[0, 0, 0, (1.0 / (self.omega**3))*(np.sinh(self.omega*T) - T), (1.0 / (self.omega**2))*(np.cosh(self.omega*T) - 1.0), (1.0 / self.omega)*np.sinh(self.omega*T), 0, 0, 0]]).T
        M_col3 = np.array([[0, 0, 0, 0, 0, 0, (1.0 / (self.omega**3))*(np.sinh(self.omega*T) - T), (1.0 / (self.omega**2))*(np.cosh(self.omega*T) - 1.0), (1.0 / self.omega)*np.sinh(self.omega*T)]]).T

        B = np.hstack((M_col1, M_col2, M_col3))

        return B

    def D(self, T):

        D_row1 = np.array([[1, 0, -self.h_CoM/SystemModel._g, 0, 0, 0, 0, 0, 0]])
        D_row2 = np.array([[0, 0, 0, 1, 0, -self.h_CoM/SystemModel._g, 0, 0, 0]])

        D = np.vstack((D_row1, D_row2))

        return D
