import numpy as np
import cv2
import class_agent
import matplotlib.pyplot as plt

ratio_displ = 1
frame_height = 500
frame_width = 500

##################################
N = 25
EPS = 0.01
C1 = 0.01
C2 = 0.1
D = 25
H = 0.9 
R = 1.01*D
A = 0.5
B = 0.5

DT = 0.04

V0X = 5
V0Y = 0.0

system = class_agent.flock(dim=2, n_a=N, n_c=1, epsilon=EPS, h=H, r=R, d=D, c1=C1, c2=C2, a=A, b=B)
print("Number of agents : ", system.nb_agents)

q_init = np.random.rand(system.nb_agents, system.nb_agents)*500
p_init = np.random.rand(system.nb_agents, system.nb_agents)*25
system.set_a_agents(q_init, p_init)
system.set_c_agents(np.array([150.0, 250.0]), np.array([V0X, V0Y]))
system.update_mass_center()
print("Mass center : ", system.c_agents[0].qr)

##################################

# DRAWING FUNCTIONS

def comp_triangle(q, p):
    v1 = p
    v2 = np.array([-p[1], p[0]])
    e1 = v1/np.linalg.norm(v1)*5
    e2 = v2/np.linalg.norm(v2)*5
    p1 = q + 2*e1
    p2 = q + 0.7*e2
    p3 = q - 0.7*e2
    return p1, p2, p3
  
def draw_triangle(img, p1, p2, p3, ratio, color=(255, 0, 0), center=(100, 100)):
    p1 = p1-center+250
    p2 = p2-center+250
    p3 = p3-center+250
    cv2.line(img, (int(p1[0]*ratio), int(p1[1]*ratio)), (int(p2[0]*ratio), int(p2[1]*ratio)), color, 1) 
    cv2.line(img, (int(p2[0]*ratio), int(p2[1]*ratio)), (int(p3[0]*ratio), int(p3[1]*ratio)), color, 1) 
    cv2.line(img, (int(p1[0]*ratio), int(p1[1]*ratio)), (int(p3[0]*ratio), int(p3[1]*ratio)), color, 1) 

def write_image(img, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(img, text, pos, font, fontScale=0.3, color=(0, 0, 0), thickness=1)

def draw_axis(img, center, size=100, im_pos=250):
    x, y = int(center[0]), int(center[1])
    cv2.line(img, (im_pos-size,im_pos-size), (im_pos+size, im_pos-size), color=(0,0,0))
    cv2.line(img, (im_pos-size,im_pos-size), (im_pos-size, im_pos+size), color=(0,0,0))
    #cv2.circle(img, (im_pos,im_pos), 1, (0,0,0), thickness=2)
    write_image(img, f"{x-size}", (im_pos-size, im_pos-size-5))
    write_image(img, f"{y-size}", (im_pos-size-20, im_pos-size+10))
    for i in range(1,2*size//50+1):
        write_image(img, f"{x-size+i*50}", (im_pos-size+i*50, im_pos-size-5))
        write_image(img, f"{y-size+i*50}", (im_pos-size-20, im_pos-size+i*50))

n_ite = 2000

# GRAPH
deviation_energy = []
velocity_miss_match = []
coes_radius = []
connectivity = []
time = [i*DT for i in range(n_ite)]


# SYSTEM SIMULATION 
for i in range(n_ite):

    system.update_c_agent(DT)
    system.update_a_agent(DT)
    system.update_mass_center()

    deviation_energy.append(system.compute_deviation_energy())
    velocity_miss_match.append(system.compute_vel_missmatch())
    coes_radius.append(system.commute_coe_radius())
    connectivity.append(system.compute_connectivity())

    # Drawing phase

    img = np.ones((frame_height, frame_width, 3), np.uint8) * 255

    write_image(img, f"Time = {i*DT:3f} [s]", (25, 25))
    draw_axis(img, system.c_agents[0].qr)

    for i, agent in enumerate(system.a_agents):
        p1, p2, p3 = comp_triangle(agent.qi, agent.pi)
        draw_triangle(img, p1, p2, p3, ratio_displ, center=system.c_agents[0].qr)
        for j, agent2 in enumerate(system.a_agents):
            if(system.adj_m[i][j]):
                cv2.line(img, (int(agent.qi[0]-system.c_agents[0].qr[0]+250), int(agent.qi[1]-system.c_agents[0].qr[1]+250)), (int(agent2.qi[0]-system.c_agents[0].qr[0]+250), int(agent2.qi[1]-system.c_agents[0].qr[1]+250)), (0,255,0), 1)

    for agent in system.c_agents:
        p1, p2, p3 = comp_triangle(agent.qr, agent.pr)
        draw_triangle(img, p1, p2, p3, ratio_displ, (0,0,255), center=system.c_agents[0].qr)

    cv2.imshow("FLOCK IT", img) 

    cv2.waitKey(1)


####################################
####################################
# Deviation energy
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot 1
axs[0, 0].plot(time, deviation_energy, label='Deviation Energy')
axs[0, 0].set_xlabel('time [s]')
axs[0, 0].set_ylabel(r'E(q)/$d^2$')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2
axs[0, 1].plot(time, velocity_miss_match, label='Velocity Miss-match')
axs[0, 1].set_xlabel('time [s]')
axs[0, 1].set_ylabel('K(v)/n')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 3
axs[1, 0].plot(time, coes_radius, label='Cohesion Radius')
axs[1, 0].set_xlabel('time [s]')
axs[1, 0].set_ylabel('R(q)')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot 4
axs[1, 1].plot(time, connectivity, label='Connectivity')
axs[1, 1].set_xlabel('time [s]')
axs[1, 1].set_ylabel('C(q)')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout for better appearance
plt.tight_layout()

# Display the plots
plt.savefig(f"graphs/n={N}_t={n_ite}.pdf", bbox_inches='tight')
plt.show()






