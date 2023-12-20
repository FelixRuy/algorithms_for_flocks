import numpy as np
import model_functions as fnc

# alpha agent class
class a_agent:

    def __init__(self, c1, c2, dim=2) -> None:
        self.qi = np.zeros(dim) # position
        self.pi = np.zeros(dim) # velocity
        self.ui = np.zeros(dim) # acceleration
        self.fi = lambda qi, qr, pi, pr : -c1*(qi-qr)-c2*(pi-pr)
    
# gamma agent class
class c_agent:

    def __init__(self, dim=2) -> None:
        self.qr = np.zeros(dim) # position
        self.pr = np.zeros(dim) # velocity
        self.fr = lambda q, p : 0 # evolution of the velocity : -c1(qi-qr)-c2(pi-pr)

    def set_fr(self, f):
        self.fr = f

# flock system
class flock:

    def __init__(self, dim, n_a, n_c, epsilon, r, d, h=0.5, c1=0.01, c2=0.1, a=0.5, b=0.5) -> None:
        self.dimensions = dim
        self.nb_agents = n_a
        self.eps = epsilon
        self.r = r
        self.d = d
        self.h = h
        self.a = a
        self.b = b
        self.a_agents = [a_agent(c1, c2, dim) for _ in range(n_a)]
        self.c_agents = [c_agent(dim) for _ in range(n_c)]
        self.adj_m = np.zeros((n_a,n_a))
        self.mass_center = np.zeros(2)
        self.velocity_average = np.zeros(2)


    def set_a_agents(self, q, p) -> None :
        for i, a_a in enumerate(self.a_agents):
            #print(a_a.qi)
            for d in range(self.dimensions):
                a_a.qi[d] = q[d][i]
                a_a.pi[d] = p[d][i]
    
    def set_c_agents(self, qd, pd) -> None :
        # Only one c agent for now
        if(len(self.c_agents)>0):
            c_a = self.c_agents[0]
            c_a.qr = qd
            c_a.pr = pd
            c_a.fr = lambda qr, pr : [0, 0]
        else: print("No c-agents to set : n_c size is zero.")
        #c_a.fr = lambda qr, pr : [0, np.sin(np.pi*2*qr[0])*0.03] 

    # update the position of the a agents
    def update_a_agent(self, dt):
        if (len(self.c_agents)>0) : c_a = self.c_agents[0]
        for i,a_a1 in enumerate(self.a_agents):
            if (len(self.c_agents)>0) : a_a1.pi += dt * a_a1.fi(a_a1.qi, c_a.qr, a_a1.pi, c_a.pr)
            for j,a_a2 in enumerate(self.a_agents):
                if(a_a1 != a_a2):
                    a_a1.pi += dt * fnc.phi_a(fnc.norm_sigm(a_a2.qi-a_a1.qi, self.eps), self.r, self.d, self.eps, self.h, a=self.a, b=self.b) * fnc.comp_n_ij(a_a1.qi, a_a2.qi, self.eps)
                    self.adj_m[i][j] = fnc.comp_a_ij(a_a1.qi, a_a2.qi, self.r, self.eps)
                    a_a1.pi += dt * self.adj_m[i][j] * (a_a2.pi-a_a1.pi)

        for a_a1 in self.a_agents:
            a_a1.qi += dt * a_a1.pi

    # update the position of the c agent
    def update_c_agent(self, dt):
        if(len(self.c_agents)>0):
            c_ag = self.c_agents[0]
            c_ag.qr += c_ag.pr * dt
            c_ag.pr += c_ag.fr(c_ag.qr, c_ag.pr)
        else: print("No c-agents to update : n_c size is zero.")

    def update_mass_center(self):
        x = 0
        y = 0
        vx = 0
        vy = 0
        for a1 in self.a_agents:
            x += a1.qi[0]
            y += a1.qi[1]
            vx += a1.pi[0]
            vy += a1.pi[1]
        self.mass_center[0] = x/self.nb_agents
        self.mass_center[1] = y/self.nb_agents
        self.velocity_average[0] = vx/self.nb_agents
        self.velocity_average[1] = vy/self.nb_agents

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
        


    def commute_coe_radius(self):
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

        