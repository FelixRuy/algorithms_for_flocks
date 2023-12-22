import numpy as np
import display
import cv2
import class_agent_peer
import matplotlib.pyplot as plt

ratio_displ = 1
frame_height = 500
frame_width = 500

DISPLAY_OBJECTIVE=False # True to display each individual obj (gamma agent)
DISPLAY_BETA_AGENT=False # True to display each beta agent
SHOW_PLOT=True # True to plot the deviation energy, velocity mismatch, cohesion radius, connectivity

OPTION=3 # 1 for algorithm 1, 2 for alg 2, 3 for alg 3

N = 25 # number of alpha agents
EPS = 0.01

C1 = 0.05 # group obj position
C2 = 0.1 # group obj velocity
D = 25 # distance between alpha agents
H = 0.2
R = 1.01*D
A = 5 # parameters for phi_a
B = 10

C1O = 0.5 # obstacle position
C2O = -1 # obstacle velocity
C1A = 0.2 # other a agents position
C2A = 0.8 # other a agents velocity
DO = 10 # distance with obstacles
RO = DO*1.01
HO = 0.9

DT = 0.04 # time step

V0X = 5.0 # initial velocity of objective x
V0Y = 0.0 # initial velocity of objective y

obstacle1 = (np.array([375, 200]), 10.0) # obstacles (pos, radius)
obstacle2 = (np.array([475, 250]), 20.0)
obstacle3 = (np.array([450, 300]), 10.0)
obstacle4 = (np.array([350, 300]), 20.0)
obstacle5 = (np.array([550, 250]), 10.0)
system = class_agent_peer.flock(dim=2, n_a=N, epsilon=EPS, h=H, r=R, d=D, c1=C1, c2=C2, a=A, b=B, option=OPTION, obstacles=[obstacle1, obstacle2, obstacle3, obstacle4, obstacle5], do=DO, ro=RO, c1a=C1A, c2a=C2A, c1o=C1O, c2o=C2O)
print("Number of agents : ", system.nb_agents)

q_init = np.random.rand(system.nb_agents, system.dimensions)*200+100 # initial position of alpha agents
p_init = np.random.rand(system.nb_agents, system.dimensions) # initial velocity of alpha agents
system.set_a_agents(q_init, p_init)
if(system.option != 1):
    qr = np.zeros((N,2))
    pr = np.zeros((N,2))
    for i in range(N):
        qr[i] = np.array([150.0, 250.0])
        pr[i] = np.array([V0X, V0Y])
    system.set_c_agents(qr, pr)
system.update_mass_center() # mass center for moving frame to compute stability criter
print("Mass center : ", system.mass_center) 

##################################

if (OPTION == 1 or OPTION == 2) : n_ite = 1000
elif (OPTION == 3) : n_ite = 2500
else : raise "BAD OPTION SELECTED"

# GRAPH
if SHOW_PLOT:
    deviation_energy = []
    velocity_miss_match = []
    coes_radius = []
    connectivity = []
    time = [i*DT for i in range(n_ite)]


# SYSTEM SIMULATION 
for m in range(n_ite):

    if(system.option != 1) : system.update_c_agent(DT)
    system.update_a_agent(DT)
    system.update_mass_center()

    if(system.option == 1) : frame_center = np.array([250, 250])
    else :
        frame_center = np.zeros(2)
        for ac in system.c_agents:
            frame_center += ac.qr / system.nb_agents

    if SHOW_PLOT:
        deviation_energy.append(system.compute_deviation_energy())
        velocity_miss_match.append(system.compute_vel_missmatch())
        coes_radius.append(system.compute_coe_radius())
        connectivity.append(system.compute_connectivity())

    # Drawing phase

    img = np.ones((frame_height, frame_width, 3), np.uint8) * 255

    display.write_image(img, f"Time = {m*DT:3f} [s]", (25, 25))
    display.draw_axis(img, frame_center)

    for i, agent in enumerate(system.a_agents):
        p1, p2, p3 = display.comp_triangle(agent.qi, agent.pi)
        display.draw_triangle(img, p1, p2, p3, ratio_displ, center=frame_center)
        for j, agent2 in enumerate(system.a_agents):
            if(system.adj_m[i][j]):
                cv2.line(img, (int(agent.qi[0]-frame_center[0]+250), int(agent.qi[1]-frame_center[1]+250)), (int(agent2.qi[0]-frame_center[0]+250), int(agent2.qi[1]-frame_center[1]+250)), (0,255,0), 1)

    if(system.option != 1 and DISPLAY_OBJECTIVE):
        for agent in system.c_agents:
            p1, p2, p3 = display.comp_triangle(agent.qr, agent.pr)
            display.draw_triangle(img, p1, p2, p3, ratio_displ, (0,0,255), center=frame_center) 

    if(system.option == 3):
        for k, obs in enumerate(system.obstacles):
            display.draw_circle(img, obs.yk, obs.Rk, center=frame_center)
            if DISPLAY_BETA_AGENT:
                for i, agent in enumerate(system.b_agents[k]):
                    p1, p2, p3 = display.comp_triangle(agent.qk, agent.pk)
                    display.draw_triangle(img, p1, p2, p3, ratio_displ, (255, 0, 255), center=frame_center)
                    cv2.line(img, (int(agent.qk[0]-frame_center[0]+250), int(agent.qk[1]-frame_center[1]+250)), (int(system.a_agents[i].qi[0]-frame_center[0]+250), int(system.a_agents[i].qi[1]-frame_center[1]+250)), (255,0,255), 1)
    cv2.imshow("FLOCK IT", img)

    cv2.waitKey(1)



####################################
####################################
if SHOW_PLOT:

    time = [i*0.04 for i in range(n_ite)]

    fig, axs = plt.subplots(2, 2, figsize=(9, 4))

    # Plot 1
    axs[0,0].plot(time, deviation_energy, color="seagreen", label=r'$c_1, c_2$ = ({}, {})'.format(C1, C2))
    # Plot 2
    axs[0,1].plot(time, velocity_miss_match, color="seagreen", label=r'$c_1, c_2$ = ({}, {})'.format(C1, C2))
    # Plot 3
    axs[1,0].plot(time, coes_radius, color="seagreen", label=r'$c_1, c_2$ = ({}, {})'.format(C1, C2))
    # Plot 4
    axs[1,1].plot(time, connectivity, color="seagreen", label=r'$c_1, c_2$ = ({}, {})'.format(C1, C2))

    axs[0,0].set_xlabel('time [s]')
    axs[0,0].set_ylabel(r'E(q)/$d^2$')
    axs[0,0].legend(title="Deviation Energy")
    axs[0,0].grid(True)

    axs[0,1].set_xlabel('time [s]')
    axs[0,1].set_ylabel(r'K(v)/n')
    axs[0,1].legend(title="Velocity Mismatch")
    axs[0,1].grid(True)

    axs[1, 0].set_xlabel('time [s]')
    axs[1, 0].set_ylabel(r'R(q)')
    axs[1, 0].legend(title="Cohesion Radius")
    axs[1, 0].grid(True)

    axs[1, 1].set_xlabel('time [s]')
    axs[1, 1].set_ylabel(r'C(q)')
    axs[1, 1].legend(title="Connectivity")
    axs[1, 1].grid(True)

    plt.show()