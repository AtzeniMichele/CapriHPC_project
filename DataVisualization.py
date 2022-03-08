import matplotlib.pyplot as plt

# elapsed time using 1,2,4,8 cpus for parallel computing
plt.plot([1,2,4],[1429, 830, 423],'ro--')
plt.axis([0, 9, 0, 1500])
plt.xlabel('Number of CPUs')
plt.ylabel('Elapsed time [s]')
plt.show()

#MSE mean value
plt.errorbar([1,2,4],[0.14788220337650443,0.13833703978414208,0.15133289396671432],[0.04053566136473489,0.018074495478963913,0.03061780346487201],linestyle='None',marker ='^',color='r',capsize = 3)
#plt.axis([0, 9, 0.120, 0.140])
plt.xlabel('Number of CPUs')
plt.ylabel('MSE mean (sd)')
plt.show()

#MAE mean value
plt.errorbar([1,2,4],[0.22741584782668856,0.2277953658889897,0.23273514556763036],[0.022851533457693644,0.015424730243707302,0.018499025255352467],linestyle='None',marker ='^',color='b',capsize = 3)
plt.xlabel('Number of CPUs')
plt.ylabel('MAE mean (sd)')
plt.show()

#wallclock time
plt.plot([1,2,4], [1440, 840, 420],'ro--')
plt.axis([0, 9, 0, 1500])
plt.xlabel('Number of CPUs')
plt.ylabel('Wall-Clock time [s]')
plt.show()
#
#RAM Usage
plt.plot([1,2,4], [0.86,1.24,1.98],'bo-')
plt.xlabel('Number of CPUs')
plt.ylabel('RAM Usage [GB]')
plt.show()
