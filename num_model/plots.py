import matplotlib.pyplot as plt
import numpy as np

time = [i*0.04 for i in range(1500)]

fig, axs = plt.subplots(2, 2, figsize=(13, 10))

color_i = 0
colors = ["seagreen", "skyblue", "darkblue", "mediumpurple", "gray"]

for C1, C2 in [(0.01, 0.1), (0.05, 0.1), (0.01, 0.2), (0.007, 0.1), (0.015, 0.1)]:

    deviation_energy = np.loadtxt(f"data/c1={C1}_c2={C2}_deviation_energy.txt")
    velocity_miss_match = np.loadtxt(f"data/c1={C1}_c2={C2}_velocity_miss_match.txt")
    coes_radius = np.loadtxt(f"data/c1={C1}_c2={C2}_coes_radius.txt")
    connectivity = np.loadtxt(f"data/c1={C1}_c2={C2}_connectivity.txt")

    # Plot 1
    axs[0, 0].plot(time, deviation_energy, color=colors[color_i], label=r'$c_1, c_2$ = ({}, {})'.format(C1, C2))

    # Plot 2
    axs[0, 1].plot(time, velocity_miss_match, color=colors[color_i], label=r'$c_1, c_2$ = ({}, {})'.format(C1, C2))
    
    # Plot 3
    axs[1, 0].plot(time, coes_radius, color=colors[color_i], label=r'$c_1, c_2$ = ({}, {})'.format(C1, C2))
    
    # Plot 4
    axs[1, 1].plot(time, connectivity, color=colors[color_i], label=r'$c_1, c_2$ = ({}, {})'.format(C1, C2))
    
    color_i += 1


axs[0, 0].set_xlabel('time [s]')
axs[0, 0].set_ylabel(r'E(q)/$d^2$')
axs[0, 0].legend(title="Deviation Energy")
axs[0, 0].grid(True)

axs[0, 1].set_xlabel('time [s]')
axs[0, 1].set_ylabel('K(v)/n')
axs[0, 1].legend(title="Velocity Mismatch")
axs[0, 1].grid(True)

axs[1, 0].set_xlabel('time [s]')
axs[1, 0].set_ylabel('R(q)')
axs[1, 0].legend(title="Cohesion Radius")
axs[1, 0].grid(True)

axs[1, 1].set_xlabel('time [s]')
axs[1, 1].set_ylabel('C(q)')
axs[1, 1].legend(title="Connectivity")
axs[1, 1].grid(True)

plt.savefig("graphs_stab/choiceOfC.pdf", bbox_inches='tight')
plt.show()