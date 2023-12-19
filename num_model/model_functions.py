import numpy as np

# classic norm definition
def norm_2(z):
    return np.linalg.norm(z)

def pair_wise_potential(z):
    return z**2

# sigma norm definition
def norm_sigm(z, eps):
    return (1/eps)*(np.sqrt(1+eps*norm_2(z)**2)-1)

# bump function
def rho_h(z, h):
    if(0 <= z and z < h):
        return 1
    if(h <= z and z <= 1):
        return (1/2)*(1+np.cos(np.pi*((z-h)/(1-h))))
    else:
        return 0

# phi function : uneven sigmoidale function 
def phi(z,a,b):
     a = 0.005
     b = 0.2
     c = abs(a-b)/np.sqrt(4*a*b)
     sigma1 = lambda x : x/np.sqrt(1+x*x)
     return (1/2)*((a+b)*sigma1(z+c)+(a-b))

# phi_alpha function : pair wise potential with finite cut-off
def phi_a(z, r, d, eps, h=1, a=0.05, b=0.05):
    r_a = norm_sigm(r, eps)
    d_a = norm_sigm(d, eps)
    return rho_h(z/r_a, h)*phi(z-d_a, a, b)

# Constructor of the adj matrix 
def comp_a_ij(qi, qj, r, eps, h=1):
    r_a = norm_sigm(r, eps)
    q = qi-qj
    return rho_h(norm_sigm(q, eps)/r_a, h)

# vector along the line connection qi, qj
def comp_n_ij(qi, qj, eps):
    return (qj-qi)/np.sqrt(1+eps*norm_2(qj-qi)**2)




