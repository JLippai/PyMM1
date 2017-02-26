"""
A simple simulation tool for generating sample paths of an M|M|1 queueing system

The program takes arrival rate lambda, departure rate mu, CONSTRAINT_MODE, maximum 
simulation runtime, and PLOT as parameters. CONSTRAINT_MODE dictates whether the run time is
specified in time units (0) or number of departures (1). The program will plot the generated 
sample path if PLOT is set. The output always includes the calculated average queue length and 
average system time.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import heapq

MAXTIME = 100000
MAXDEPARTS = 5000

CONSTRAINT_MODE = 0

Lambda = 1.0
Mu = 2.0

if len(sys.argv) > 4:
	Lambda = float(sys.argv[1])
	Mu = float(sys.argv[2])
	CONSTRAINT_MODE = int(sys.argv[3])
	if CONSTRAINT_MODE == 0:
		MAXTIME = int(sys.argv[4])
	else:
		CONSTRAINT_MODE = 1
		MAXDEPARTS = int(sys.argv[4])

rho = Lambda/Mu

rates = {'a':Lambda, 'd':Mu}
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
	while (True):	
		new_event = heapq.heappop(EVENTHEAP)
		if (new_event[1] == 'd'):
			departureCount += 1
		updateState(new_event, queues)
		updateFeasibleEvents(new_event, times)

		if (CONSTRAINT_MODE):
			if (departureCount >= MAXDEPARTS):
				break
		else:
			if (times[-1] >= MAXTIME):
				break

	tarray = np.array(times)
	qarray = np.array(queues)
	q_substantive = qarray[:-1]
	difft = np.diff(tarray)
	u = np.sum(q_substantive*difft)
	L = u/tarray[-1]
	S = u/len(arrivals)
	return tarray, qarray, L, S

def main():
	tarray, qarray, L, S = runSimulation()
	print("lambda = %.1f,    mu = %.1f,    rho = %.4f\nAvg queue Avg sys time\n%.6f, %.6f"%(Lambda, Mu, rho, L, S))
	if (len(sys.argv) > 5 and sys.argv[5]) == '1':
		plt.bar(tarray, qarray, edgecolor="none")
		plt.xlabel("Time")
		plt.ylabel("Queue length")
		plt.xlim(0, tarray[-1])
		plt.savefig("samplepath.png")
		plt.close()
	
if __name__ == '__main__':
	main()
