'''
Geman-McClure loss function
robustness — that a model be less influenced by outliers than by inliers [18, 20].

refer to
1. A General and Adaptive Robust Loss Function, Jonathan T. Barron, 2018
2. Chapter 2 M-Estimators and Half-Quadratic Minimization

3. GMOf in SMPLify
'''

import numpy as np
import torch
import matplotlib.pyplot as plt

def GMOf(x, sigma):
    '''
        x: (B, N, 3)
        sigma (float) scale parameter

        2*(x/sigma)**2 / ((x/sigma)**2 + 4)
    '''
    squared_input = x**2
    return 2*squared_input / (squared_input + 4*sigma**2)

if __name__ == '__main__':   
    a = torch.zeros(2, 6, 3)
    b = torch.ones(2,6,3)

    # loss_fun = lambda a, b: (a-b)**2
    # loss_fun = lambda a, b: abs(a-b)
    loss_fun = lambda a, b: GMOf(a-b,1)

    dif = loss_fun(a,b)
    dif = dif.sum(2)

    print('finished')

    x = np.linspace(-6, 6, 100)
    plt.plot(x, abs(x), label='abs')
    #plt.plot(x, x**2, label='quadratic')
    plt.plot(x, GMOf(x,1), label='GM')

    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("Simple Plot")
    plt.legend()

    plt.show()

# #!/usr/bin/env python
# import numpy as np
# import scipy
# import scipy.sparse as sp
# from chumpy import Ch

# __all__ = ['GMOf']


# def GMOf(x, sigma):
#     """Given x and sigma in some units (say mm),
#     returns robustified values (in same units),
#     by making use of the Geman-McClure robustifier."""

#     result = SignedSqrt(x=GMOfInternal(x=x, sigma=sigma))
#     return result


# class SignedSqrt(Ch):
#     dterms = ('x', )
#     terms = ()

#     def compute_r(self):
#         return np.sqrt(np.abs(self.x.r)) * np.sign(self.x.r)

# class GMOfInternal(Ch):
#     dterms = 'x', 'sigma'

#     def on_changed(self, which):
#         if 'sigma' in which:
#             assert (self.sigma.r > 0)

#         if 'x' in which:
#             self.squared_input = self.x.r**2.

#     def compute_r(self):
#         return (self.sigma.r**2 *
#                 (self.squared_input /
#                  (self.sigma.r**2 + self.squared_input))) * np.sign(self.x.r)


# if __name__ == '__main__':            
#     obj_joints3d = lambda w, sigma: (w * GMOf((joints3D - model.J_transformed), sigma))
#     obj_vertices = lambda w, sigma: (w * GMOf(((vertices.T * weights_corr)
#                                      - (model.T * weights_corr)).T, sigma))

#     objs['j2d'] = obj_j2d(1., 100)
#     objs['j2d'].r**2

#     print('\terror(vertices) = %.2f' % (obj_vertices(100, 100).r**2).sum())