import numpy as np
import display
import cv2
import class_agent
import matplotlib.pyplot as plt

ratio_displ = 1
frame_height = 500
frame_width = 500

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

system = class_agent.flock(dim=2, n_a=N, n_c=0, epsilon=EPS, h=H, r=R, d=D, c1=C1, c2=C2, a=A, b=B)
print("Number of a agents : ", system.nb_agents)
print("Number of a agents : ", 0)

q_init = np.random.rand(system.nb_agents, system.nb_agents)*400
p_init = np.random.rand(system.nb_agents, system.nb_agents)*25
system.set_a_agents(q_init, p_init)
system.update_mass_center()
print("Mass center : ", system.mass_center)

##################################

n_ite = 1500

# GRAPH
deviation_energy = []
velocity_miss_match = []
coes_radius = []
connectivity = []
time = [i*DT for i in range(n_ite)]


# SYSTEM SIMULATION 
for k in range(n_ite):

    system.update_a_agent(DT)
    system.update_mass_center()

    deviation_energy.append(system.compute_deviation_energy())
    velocity_miss_match.append(system.compute_vel_missmatch())
    coes_radius.append(system.commute_coe_radius())
    connectivity.append(system.compute_connectivity())

    # Drawing phase

    img = np.ones((frame_height, frame_width, 3), np.uint8) * 255

    display.write_image(img, f"Time = {k*DT:3f} [s]", (25, 25))
    display.draw_axis(img, system.mass_center)

    for i, agent in enumerate(system.a_agents):
        p1, p2, p3 = display.comp_triangle(agent.qi, agent.pi)
        display.draw_triangle(img, p1, p2, p3, ratio_displ, center=system.mass_center)
        for j, agent2 in enumerate(system.a_agents):
            if(system.adj_m[i][j]):
                cv2.line(img, (int(agent.qi[0]-system.mass_center[0]+250), int(agent.qi[1]-system.mass_center[1]+250)), (int(agent2.qi[0]-system.mass_center[0]+250), int(agent2.qi[1]-system.mass_center[1]+250)), (0,255,0), 1)

    cv2.imshow("FLOCK IT", img)

    if(k in [100, 700, 1000]):
        cv2.imwrite(f"graphs_flock/A1--n={N}_t={k*DT}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    cv2.waitKey(1)


####################################
####################################
# Deviation energy
fig, axs = plt.subplots(1, 2, figsize=(6, 3))

""" # Plot 1
axs[0, 0].plot(time, deviation_energy, color="seagreen", label='Deviation Energy')
axs[0, 0].set_xlabel('time [s]')
axs[0, 0].set_ylabel(r'E(q)/$d^2$')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2
axs[0, 1].plot(time, velocity_miss_match, color="seagreen", label='Velocity Miss-match')
axs[0, 1].set_xlabel('time [s]')
axs[0, 1].set_ylabel('K(v)/n')
axs[0, 1].legend()
axs[0, 1].grid(True) """

# Plot 3
axs[0].plot(time, coes_radius, color="seagreen", label='Cohesion Radius')
axs[0].set_xlabel('time [s]')
axs[0].set_ylabel('R(q)')
axs[0].legend()
axs[0].grid(True)

# Plot 4
axs[1].plot(time, connectivity, color="seagreen", label='Connectivity')
axs[1].set_xlabel('time [s]')
axs[1].set_ylabel('C(q)')
axs[1].legend()
axs[1].grid(True)

# Adjust layout for better appearance
plt.tight_layout()

# Display the plots
plt.savefig(f"graphs_stab/A1--n={N}_t={n_ite}.pdf", bbox_inches='tight')
plt.show()






