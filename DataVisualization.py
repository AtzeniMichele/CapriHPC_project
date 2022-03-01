import matplotlib.pyplot as plt

# elapsed time using 1,2,4,8 cpus for parallel computing
plt.plot([1,2,4,8],[111, 101, 66, 34],'ro')
plt.axis([0, 9, 0, 120])
plt.xlabel('Number of CPUs')
plt.ylabel('Elapsed time [s]')
plt.show()

#MSE mean value
plt.errorbar([1,2,4,8],[0.12883458297796552, 0.12715138152343763, 0.13355227139543216, 0.1319459217099502],[0.017791448420769004,0.021731807905067593,0.02383135967412193,0.028117900490724778],linestyle='None',marker ='^',color='r',capsize = 3)
#plt.axis([0, 9, 0.120, 0.140])
plt.xlabel('Number of CPUs')
plt.ylabel('MSE mean (sd)')
plt.show()

#MAE mean value
plt.errorbar([1,2,4,8],[0.2412873383947008, 0.23610423270881378, 0.2410540344109542, 0.2404229220016229],[0.017851388969616567,0.020808020951085585,0.0210351619850076,0.021785963808881502],linestyle='None',marker ='^',color='b',capsize = 3)
plt.xlabel('Number of CPUs')
plt.ylabel('MAE mean (sd)')
plt.show()

