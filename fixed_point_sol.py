import warnings
import numpy as np
import scipy as sp


#########################################################################
#
# isotopic features
#
#########################################################################

def v_phi_lam(phi, lam, a=1):
    '''
    The unique solution v for fixed-point equation
        1 / v(-lam;phi) = lam + phi * int r / 1 + r * v(-lam;phi) dH(r)
    where H is the distribution of eigenvalues of Sigma.
    For isotopic features Sigma = a*I, the solution has a closed form, which reads that
        lam>0:
            v(-lam;phi) = (-(phi+lam/a-1)+np.sqrt((phi+lam/a-1)**2+4*lam/a))/(2*lam)
        lam=0, phi>1
            v(-lam;phi) = 1/(a*(phi-1))
    and undefined otherwise.
    '''
    assert a>0
    
    min_lam = -np.inf# -(1 - np.sqrt(phi))**2 * a
    if phi<=0. or lam<min_lam:
        raise ValueError("The input parameters should satisfy phi>0 and lam>=min_lam.")
    
    if phi==np.inf:
        return 0
    elif lam!=0:
        return (-(phi+lam/a-1)+np.sqrt((phi+lam/a-1)**2+4*lam/a))/(2*lam)
    elif phi<=1.:
        return np.inf
    else:
        return 1/(a*(phi-1))
    

def vb_phi_lam(phi, lam, a=1, v=None):    
    if lam==0:
        if phi>1:
            return 1/(phi-1)
        else:
            return phi/(phi-1)
    else:
        if v is None:
            v = v_phi_lam(phi,lam,a)
        return phi/(1/a+v)**2/(
            1/v**2 - phi/(1/a+v)**2)
    
    
def vv_phi_lam(phi, lam, a=1, v=None):
    if lam==0:
        if phi>1:
            return phi/(a**2*(phi-1)**3)
        else:
            return np.inf
    else:
        if v is None:
            v = v_phi_lam(phi,lam,a)
        return 1./(
            1/v**2 - phi/(1/a+v)**2)
    
    
def tv_phi_lam(phi, phi_s, lam, v=None):
    if v is None:
        v = v_phi_lam(phi_s,lam)
        
    if v==np.inf:
        return phi/(1-phi)
    elif lam==0. and phi>1:
        return phi/(phi_s**2 - phi)
    else:
        if v>1e-3:
            tmp = phi/(1+v)**2
            tv = tmp/(1/v**2 - tmp)
        else:
            tmp = phi * v**2/(1+v)**2
            tv = tmp/(1 - tmp)
        
        return tv
    

def tc_phi_lam(phi_s, lam, v=None):
    if v is None:
        v = v_phi_lam(phi_s,lam)
    if v==np.inf:
        return 0.
    elif lam==0 and phi_s>1:
        return (phi_s - 1)**2/phi_s**2
    else:
        return 1/(1+v)**2
    

def vb_lam_phis_phi(lam, phi_s, phi, v=None):    
    if lam==0 and phi_s<=1:
        return 1+phi_s/(phi_s-1)
    else:
        if v is None:
            v = v_phi_lam(phi_s,lam)
        vsq_inv = 1/v**2
        return vsq_inv/(vsq_inv - phi/(1+v)**2)
    
    
def vv_lam_phis_phi(lam, phi_s, phi, v=None):
    if lam==0 and phi_s<=1:
        return np.inf
    else:
        if v is None:
            v = v_phi_lam(phi_s,lam)
        return phi/(
            1/v**2 - phi/(1+v)**2)



#########################################################################
#
# non-isotopic features
#
#########################################################################
    
def v_general(phi, lam, Sigma=None, v0=None):
    if Sigma is None:
        return v_phi_lam(phi, lam)
    else:
        p = Sigma.shape[0]
        
        if phi==np.inf:
            return 0
        elif lam==0 and phi<1:
            return np.inf
        elif lam<=0 and phi==1:
            return np.inf

        if v0 is None:
            v0 = v_phi_lam(phi, np.maximum(lam, -(-1 + np.sqrt(phi))**2 + 1e-6))

        # if lam<0:
        #     func = lambda v_inv,phi: (lam + phi * np.trace(np.linalg.solve(v_inv * np.identity(p) + Sigma, v_inv * Sigma)) / p)
        #     v0 = 1/v0
        # else:
        func = lambda v,phi: 1/(lam + phi * np.trace(np.linalg.solve(np.identity(p) + v * Sigma, Sigma)) / p)        
            
        # func = lambda v,phi: 1/(lam + phi * np.trace(np.linalg.solve(np.identity(p) + v * Sigma, Sigma)) / p)
        
        v = np.ndarray.item(fixed_point(func, v0, args=(phi,), xtol=1e-3, maxiter=int(1e4)))

        # if lam<0:
        #     v = 1/v

        return v


def tv_general(phi, phi_s, lam, Sigma=None, v=None, ATA=None):
    if lam==0 and phi_s<1:
        return phi/(1 - phi)
    if Sigma is None:
        return tv_phi_lam(phi, phi_s, lam, v)
    else:
        if v is None:
            v = v_general(phi_s, lam, Sigma)
        if v==np.inf:
            return phi/(1-phi)
        if ATA is None:
            ATA = Sigma

        v_inv = 1/v
        p = Sigma.shape[0]
        tmp = phi * np.trace(            
                np.linalg.solve(
                    v_inv * np.identity(p) + Sigma, 
                    np.linalg.solve(v_inv * np.identity(p) + Sigma, v_inv**2 * ATA @ Sigma)
                )
        ) / p
        tv = tmp/(v_inv**2 - tmp)
        return tv


# def tc_general(phi_s, lam, Sigma=None, beta=None, v=None):
#     if lam==0 and phi_s<1:
#         return 0
#     if Sigma is None:
#         return tc_phi_lam(phi_s, lam, v)
#     else:
#         if v is None:
#             v = v_general(phi_s, lam, Sigma)
#         if v==np.inf:
#             return 0.
#         p = Sigma.shape[0]
#         tmp = np.linalg.solve(np.identity(p) + v * Sigma, beta[:,None])
#         tc = np.trace(tmp.T @ Sigma @ tmp)
#         return tc
    
    
def tc_general(phi, phi_s, lam, Sigma=None, beta=None, 
               v=None, tv=None, ATA=None):
    if lam==0 and phi_s<1:
        return 0
    if Sigma is None:
        return tc_phi_lam(phi_s, lam, v)
    else:
        if v is None:
            v = v_general(phi_s, lam, Sigma)            
        if v==np.inf:
            return 0.
        if ATA is None:
            ATA = Sigma
        if tv is None:
            tv = tv_general(phi, phi_s, lam, Sigma, v, ATA)
        
        v_inv = 1/v

        p = Sigma.shape[0]
        tmp = np.linalg.solve(v_inv * np.identity(p) + Sigma, v_inv * beta[:,None])
        tc = np.trace(tmp.T @ (tv * Sigma + ATA) @ tmp)
        return tc





#########################################################################
#
# lam_min
#
#########################################################################
from scipy.optimize import fixed_point


def lam_min(phi):
    if phi==np.inf:
        return 0
    v = 1/(-1 + np.sqrt(phi))
    lam = 1/v - phi / (1 + v)

    return lam


def lam_min_general(phi, Sigma=None, r_min=None):

    if Sigma is None:    
        return lam_min(phi)
    else:
        p = Sigma.shape[0]
        
        if phi==np.inf:
            return 0
        elif phi==1.:
            return 0.      
        
        if r_min is None:
            try:
                r_min = sp.sparse.linalg.eigsh(Sigma, k=1, which='SM', return_eigenvectors=False)[0]
            except:
                r_min = np.linalg.eigh(Sigma)[0][0]
        v0 = 1 / ((-1 + np.sqrt(phi)) * r_min)

        func = lambda v,phi: (-1)**(phi<1)/np.sqrt(phi * np.trace(            
                np.linalg.solve(
                    np.identity(p) + v * Sigma, 
                    np.linalg.solve(np.identity(p) + v * Sigma, Sigma @ Sigma)
                )
        ) / p)
        v = fixed_point(func, v0, args=(phi,))
        # func = lambda v_inv,phi: (-1)**(phi<1) * np.sqrt(phi * np.trace(            
        #         np.linalg.solve(
        #             v_inv * np.identity(p) + Sigma, 
        #             np.linalg.solve(v_inv * np.identity(p) + Sigma, v_inv**2 * Sigma @ Sigma)
        #         )
        # ) / p)
        v_inv = 1/fixed_point(func, v0, args=(phi,))
        assert v_inv > -r_min
        
        # lam = 1/v - phi * np.trace(np.linalg.solve(np.identity(p) + v * Sigma, Sigma)) / p
        lam = v_inv - phi * np.trace(np.linalg.solve(v_inv * np.identity(p) + Sigma, v_inv * Sigma)) / p

        return lam