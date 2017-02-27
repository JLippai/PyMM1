"""
Usage: python mm1.py [LAMBDA=1.0 MU=2.0 LIMIT_SWITCH=0 LIMIT_VALUE=10000 FIGURE_SAVE=1]

The program simulates an M|M|1 queueing system with arrival rate LAMBDA and departure rate MU. 
LIMIT_SWITCH=0 will allow defining the runtime LIMIT_VALUE in time units; LIMIT_SWITCH=1 will 
define the runtime LIMIT_VALUE in departures. Setting FIGURE_SAVE will result in plots of a 
sample path of queue length and of estimators average queue length and average system time being 
generated and saved to samplepath.png and estimators.png
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import heapq

LAMBDA = 1.0
MU = 2.0
LIMIT_SWITCH = 0
LIMIT_VALUE = 10000
FIGURE_SAVE = 1


if len(sys.argv) > 4:
	LAMBDA = float(sys.argv[1])
	MU = float(sys.argv[2])
	LIMIT_SWITCH = int(sys.argv[3])
	if LIMIT_SWITCH == 0:
		LIMIT_VALUE = int(sys.argv[4])
	else:
		LIMIT_SWITCH = 1
		LIMIT_VALUE = int(sys.argv[4])
	if len(sys.argv) > 5:
		FIGURE_SAVE = int(sys.argv[5])

rho = LAMBDA/MU

rates = {'a':LAMBDA, 'd':MU}
infeasibleEvents = {'all':['INIT'], 0:['d']}
feasibleEvents = {'all':['a', 'd']}

QUEUE = 0
EVENTHEAP = [(0, 'INIT'),]

arrivals = []
departures = []

def newLifetime(event):
	"""Generate a new event's lifetime, with the Poisson parameter specified in the rates map"""
	return -1/rates[event]*np.log(1 - np.random.rand())

def updateState(event, queues):
	"""Update the queue length as a function of current state and triggering event.
		Also updates the array of queue lengths.
	"""
	global QUEUE
	QUEUE += (event[1] == 'a') - (event[1] == 'd')
	queues.append(QUEUE)

def updateFeasibleEvents(outgoing_event, times):
	""" Update the scheduled event list given the current queue length and the event (with lifetime)
		of the event that triggered the current state. Also updates the time array times.
	"""
	global EVENTHEAP
	global arrivals
	global departures
	global feasibleEvents
	global infeasibleEvents
	try:
		a = feasibleEvents[QUEUE]
	except:
		feasibleEvents[QUEUE] = []
	try:
		a = infeasibleEvents[QUEUE]
	except:
		infeasibleEvents[QUEUE] = []

	EVENTHEAP = [(time - outgoing_event[0], event) for (time, event) in EVENTHEAP if event not in infeasibleEvents[QUEUE]]
	residualEvents = [event for (time, event) in EVENTHEAP]

	for event in ((set(feasibleEvents[QUEUE]).union(feasibleEvents['all'])) - set(residualEvents)) - (set(infeasibleEvents[QUEUE]).union(infeasibleEvents['all'])):
		heapq.heappush(EVENTHEAP, (newLifetime(event), event))
	if (times == []):
		times.append(outgoing_event[0])
	else:
		times.append(times[-1] + outgoing_event[0])
		if (outgoing_event[1] == 'a'):
			arrivals.append(times[-1])
		else:
			departures.append(times[-1])

def runSimulation():
	"""Updates the system until the maximum time or number of departures is reached,
		and returns the array of times at which events occurred, the array of the queue length
		at those times, the average queue length, and the average system time
	"""
	departureCount = 0
	times = []
	queues = []
	arrivalCountArray = [0]
	while (True):	
		new_event = heapq.heappop(EVENTHEAP)
		if (new_event[1] == 'd'):
			departureCount += 1
			arrivalCountArray.append(0)
		elif (new_event[1] == 'a'):
			arrivalCountArray.append(1)
		updateState(new_event, queues)
		updateFeasibleEvents(new_event, times)

		if (LIMIT_SWITCH):
			if (departureCount >= LIMIT_VALUE):
				break
		else:
			if (times[-1] >= LIMIT_VALUE):
				break

	tarray = np.array(times)
	qarray = np.array(queues)
	q_substantive = qarray[:-1]
	difft = np.diff(tarray)
	u = np.sum(q_substantive*difft)
	L = u/tarray[-1]
	S = u/len(arrivals)
	return tarray, qarray, arrivalCountArray, L, S

def main():
	tarray, qarray, arrivalCountArray, L, S = runSimulation()

	print("lambda = %.1f,    mu = %.1f,    rho = %.4f\nAvg queue, Avg sys time\n%.6f, %.6f"%(LAMBDA, MU, rho, L, S))

	if (FIGURE_SAVE):
		if (LIMIT_SWITCH):
			runtimeLabel = 'Departures'
		else:
			runtimeLabel = 'Time'

		uarray = np.cumsum(qarray[:-1]*np.diff(tarray))
		qdiff = np.diff(qarray)
		plt.bar(tarray, qarray, edgecolor="none")
		plt.title("Sample path of queue length vs. %s"%runtimeLabel.lower())
		plt.xlabel("%s"%runtimeLabel)
		plt.ylabel("Queue length")
		plt.xlim(0, tarray[-1])
		plt.savefig("samplepath.png")
		plt.close()

		plt.figure()
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		plt.plot(tarray[1:], uarray/tarray[1:], label=r"$\bar x$")
		plt.plot(tarray[1:], uarray/np.cumsum(arrivalCountArray[1:]), label=r"$\bar s$")
		plt.legend()
		plt.title(r"Estimators $\bar x$ and $\bar s$ as functions of %s"%runtimeLabel.lower())
		plt.xlabel("%s"%runtimeLabel)
		plt.ylabel("Estimators")
		plt.xlim(0, tarray[-1])
		plt.savefig("estimators.png")
		plt.close()

if __name__ == '__main__':
	main()
