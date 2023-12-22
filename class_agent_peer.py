import numpy as np
import model_functions as fnc

# alpha agent class
class a_agent:

    def __init__(self, c1, c2, dim=2) -> object:
        self.qi = np.zeros(dim) # position
        self.pi = np.zeros(dim) # velocity
        self.ui = np.zeros(dim) # acceleration
        self.c1 = c1
        self.c2 = c2
        self.fi = lambda qi, qr, pi, pr : np.zeros(2)

class circular_obstacle:

    def __init__(self, yk, Rk, dim=2) -> None:
        self.yk = yk
        self.Rk = Rk
        self.dim = dim

    def comp_mu(self, qi):
        return self.Rk / fnc.norm_2(qi-self.yk)
    
    def comp_P(self, qi):
        ak = (qi-self.yk)/fnc.norm_2(qi-self.yk)
        # Does it matches dimensions?? 
        return np.identity(self.dim) - ak@ak.T
    
    def comp_q_p(self, qi, pi):
        mu = self.comp_mu(qi)
        P = self.comp_P(qi)
        qik = mu*qi + (1-mu)*self.yk
        pik = mu*P@pi
        return qik, pik
    
class b_agent:

    def __init__(self) -> None:
        self.qk = np.zeros(2)
        self.pk = np.zeros(2)
    
# gamma agent class
class c_agent:

    def __init__(self, dim=2) -> object:
        self.qr = np.zeros(dim) # position
        self.pr = np.zeros(dim) # velocity
        self.fr = lambda q, p : np.zeros(2) # evolution of the velocity : -c1(qi-qr)-c2(pi-pr)

    def set_fr(self, f):
        self.fr = f

class flock:

    def __init__(self, dim=2, n_a=5, epsilon=0.01, r=1.01*25, d=25, h=0.2, c1=0.05, c2=0.1, a=2, b=4, option=2, obstacles=[], c1a=1, c2a=1, c1o=1, c2o=1, do=20, ro=1.1*20, ho=0.9) -> object:
        self.dimensions = dim
        self.nb_agents = n_a
        self.eps = epsilon
        self.r = r
        self.ro = ro
        self.d = d
        self.do = do
        self.h = h
        self.ho = ho
        self.a = a
        self.b = b
        self.c1a, self.c2a, self.c1o, self.c2o = c1a, c2a, c1o, c2o
        self.a_agents = [a_agent(c1, c2, dim) for _ in range(n_a)]
        if (option!=1) : self.c_agents = [c_agent(dim) for _ in range(n_a)]
        # useful for visualization only
        if(option==3) : self.b_agents = [[b_agent() for _ in range(n_a)] for _ in range(len(obstacles))]
        self.adj_m = np.zeros((n_a,n_a))
        self.mass_center = np.zeros(2)
        self.velocity_average = np.zeros(2)
        self.option = option
        self.obstacles = [circular_obstacle(yk, Rk) for yk, Rk in obstacles]

    def set_a_agents(self, q, p) -> None:
        # each row of q/p corresponds to an agent
        for i, aa in enumerate(self.a_agents):
            aa.qi = q[i]
            aa.pi = p[i]
            aa.fi = lambda qi, qr, pi, pr : -aa.c1*(qi-qr)-aa.c2*(pi-pr)
    
    def set_c_agents(self, q, p):
        if(self.option == 1) : return  
        # each row of q/p corresponds to an agent
        for i, ca in enumerate(self.c_agents):
            ca.qr = q[i]
            ca.pr = p[i]
            ca.fr = lambda qr, pr : np.zeros(2)
    
    def update_a_agent(self, dt):
        for i, aa1 in enumerate(self.a_agents):
            if(self.option != 1) :
                ca1 = self.c_agents[i]
                # interaction with proper gamma agent
                aa1.pi += dt * aa1.fi(aa1.qi, ca1.qr, aa1.pi, ca1.pr)
            for j, aa2 in enumerate(self.a_agents):
                # interaction with other alpha agents
                aa1.pi += self.c1a * dt * fnc.phi_a(fnc.norm_sigm(aa2.qi-aa1.qi, self.eps), self.r, self.d, self.eps, self.h, a=self.a, b=self.b) * fnc.comp_n_ij(aa1.qi, aa2.qi, self.eps)
                self.adj_m[i][j] = fnc.comp_a_ij(aa1.qi, aa2.qi, self.r, self.eps)
                aa1.pi += self.c2a * dt * self.adj_m[i][j] * (aa2.pi-aa1.pi)

            if(self.option==3):
                a_obs = 5
                b_obs = 10
                # interaction with beta agent -- obstacles
                for k,obs in enumerate(self.obstacles):
                    qik, pik = obs.comp_q_p(aa1.qi, aa1.pi)
                    grad_term = self.c1o * dt * fnc.phi_a(fnc.norm_sigm(qik-aa1.qi, eps=self.eps), self.ro, self.do, self.eps, self.ho, a_obs, b_obs) * fnc.comp_n_ij(aa1.qi, qik, self.eps)
                    aa1.pi += grad_term
                    bik = fnc.comp_a_ij(aa1.qi, qik, self.ro, self.eps, h=self.ho)
                    prox_term = self.c2o * dt * bik * (pik-aa1.pi)
                    aa1.pi += prox_term
                    self.b_agents[k][i].qk = qik
                    self.b_agents[k][i].pk = pik

        for aa1 in self.a_agents:
            aa1.qi += dt * aa1.pi

    def update_c_agent(self, dt):
        if(self.option == 1) : return None 
        for ca in self.c_agents:
            ca.pr += ca.fr(ca.qr, ca.pr) * dt
            ca.qr += ca.pr * dt

    def update_mass_center(self):
        avg_q = np.zeros(2)
        avg_p = np.zeros(2)
        for aa in self.a_agents:
            avg_q += aa.qi
            avg_p += aa.pi
        self.mass_center = avg_q/self.nb_agents
        self.velocity_average = avg_p/self.nb_agents

    def compute_deviation_energy(self):
        order_e = 0
        double_sum = 0
        for i,ai in enumerate(self.a_agents):
            for j,aj in enumerate(self.a_agents):
                if j != i : 
                    if self.adj_m[i][j] != 0:
                        order_e += 1
                        double_sum += fnc.pair_wise_potential(fnc.norm_2(aj.qi-ai.qi) - self.d)
        return 1/(1+order_e) * double_sum / self.d**2

    def compute_vel_missmatch(self):
        # miss match with the center of gravity of the flock 
        return 1/(2*self.nb_agents) * np.sum([fnc.norm_2(ai.pi-self.velocity_average)**2 for ai in self.a_agents])
        
    def compute_coe_radius(self):
        R = 0
        for ai in self.a_agents:
            r = fnc.norm_2(ai.qi - self.mass_center)
            if(r) > R: R = r
        return R

    def compute_connectivity(self):
        # C(t) = (1/n - 1) rank(L(q(t)))
        deg_mat = np.diag([np.sum(self.adj_m[i]) for i in range(self.nb_agents)])
        L = deg_mat-self.adj_m
        return 1/(self.nb_agents-1) * np.linalg.matrix_rank(L)


