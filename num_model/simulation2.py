import numpy as np
import display
import cv2
import class_agent
import matplotlib.pyplot as plt

ratio_displ = 1
frame_height = 500
frame_width = 500

# opt choice : N = 25, EPS = 0.01, C1 = 0.01, C2 = 0.2, D = 25, H = 0.5, R = 1.01*D, A = 0.5, B = 0.5
# try outs : (0.01, 0.1), (0.05, 0.1), (0.01, 0.2), (0.007, 0.1), (0.015, 0.1)

N = 50
EPS = 0.01
C1 = 0.05
C2 = 0.1
D = 25
H = 0.5
R = 1.01*D
A = 2
B = 4

DT = 0.04

V0X = 5
V0Y = 0.0

system = class_agent.flock(dim=2, n_a=N, n_c=1, epsilon=EPS, h=H, r=R, d=D, c1=C1, c2=C2, a=A, b=B)
print("Number of agents : ", system.nb_agents)

q_init = np.random.rand(system.nb_agents, system.nb_agents)*400
p_init = np.random.rand(system.nb_agents, system.nb_agents)*25
system.set_a_agents(q_init, p_init)
system.set_c_agents(np.array([150.0, 250.0]), np.array([V0X, V0Y]))
system.update_mass_center()
print("Mass center : ", system.mass_center)

##################################

n_ite = 3000

# GRAPH
deviation_energy = []
velocity_miss_match = []
coes_radius = []
connectivity = []
time = [i*DT for i in range(n_ite)]


# SYSTEM SIMULATION 
for k in range(n_ite):

    system.update_c_agent(DT)
    system.update_a_agent(DT)
    system.update_mass_center()

    deviation_energy.append(system.compute_deviation_energy())
    velocity_miss_match.append(system.compute_vel_missmatch())
    coes_radius.append(system.commute_coe_radius())
    connectivity.append(system.compute_connectivity())

    # Drawing phase

    img = np.ones((frame_height, frame_width, 3), np.uint8) * 255

    display.write_image(img, f"Time = {k*DT:3f} [s]", (25, 25))
    display.draw_axis(img, system.c_agents[0].qr)

    for i, agent in enumerate(system.a_agents):
        p1, p2, p3 = display.comp_triangle(agent.qi, agent.pi)
        display.draw_triangle(img, p1, p2, p3, ratio_displ, center=system.c_agents[0].qr)
        for j, agent2 in enumerate(system.a_agents):
            if(system.adj_m[i][j]):
                cv2.line(img, (int(agent.qi[0]-system.c_agents[0].qr[0]+250), int(agent.qi[1]-system.c_agents[0].qr[1]+250)), (int(agent2.qi[0]-system.c_agents[0].qr[0]+250), int(agent2.qi[1]-system.c_agents[0].qr[1]+250)), (0,255,0), 1)

    for agent in system.c_agents:
        p1, p2, p3 = display.comp_triangle(agent.qr, agent.pr)
        display.draw_triangle(img, p1, p2, p3, ratio_displ, (0,0,255), center=system.c_agents[0].qr)

    cv2.imshow("FLOCK IT", img)

    if(k in [2900]):
        cv2.imwrite(f"WHOUP.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    cv2.waitKey(1)


####################################
####################################
    
""" np.savetxt(f"data/c1={C1}_c2={C2}_deviation_energy.txt", np.array(deviation_energy))
np.savetxt(f"data/c1={C1}_c2={C2}_velocity_miss_match.txt", np.array(velocity_miss_match))
np.savetxt(f"data/c1={C1}_c2={C2}_coes_radius.txt", np.array(coes_radius))
np.savetxt(f"data/c1={C1}_c2={C2}_connectivity.txt", np.array(connectivity)) """


time = [i*0.04 for i in range(n_ite)]

fig, axs = plt.subplots(1, 2, figsize=(9, 4))

# Plot 1
axs[0].plot(time, deviation_energy, color="seagreen", label=r'$c_1, c_2$ = ({}, {})'.format(C1, C2))

# Plot 3
axs[1].plot(time, coes_radius, color="seagreen", label=r'$c_1, c_2$ = ({}, {})'.format(C1, C2))


axs[0].set_xlabel('time [s]')
axs[0].set_ylabel(r'E(q)/$d^2$')
axs[0].legend(title="Deviation Energy")
axs[0].grid(True)


axs[1].set_xlabel('time [s]')
axs[1].set_ylabel('R(q)')
axs[1].legend(title="Cohesion Radius")
axs[1].grid(True)


#plt.savefig("graphs_stab/ab_up.pdf", bbox_inches='tight')
plt.show()
