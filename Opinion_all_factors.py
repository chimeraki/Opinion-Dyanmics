from numpy import *
#import cPickle, gzip
from matplotlib.pyplot import *
import scipy.linalg as linalg
import scipy
import csv
import igraph
import itertools
from scipy import signal
import random
import operator
import pickle
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
import math
import copy
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph
from scipy.sparse import coo_matrix
import json
from scipy.cluster.hierarchy import dendrogram,linkage
import community

matplotlib.rc('xtick', labelsize=13) 
matplotlib.rc('ytick', labelsize=13)


random.seed(42)
#--------------------------------------
#parameters

N=100 #noofpeople
m=5 #nooftopics

#randomly initialized vectors between 0 and 1 over a random distribution.
#O= np.random.normal(0.5,0.2,(N,m))   #opinion vector
#T=np.random.normal(0.5,0.2,(N,N,m))  #trust (sensitivity)
#C= np.random.normal(0.5,0.2,(N,m))   #credibility
#I=np.random.normal(0.5,0.2,(N,m))    #influencing power
O= np.random.rand(N,m)   #opinion vector
T=np.random.rand(N,N,m)  #trust (sensitivity)
C= np.random.rand(N,m)   #credibility
I=np.random.rand(N,m)    #influencing power


G=nx.watts_strogatz_graph(N,int(N/3),0.4) # small world: (noofnodesN,each node connected to N/3 nearest neighbors, 0.4 probability of reqiring) weighted?
A=nx.to_numpy_matrix(G)            #adjacency matrix


#parameters:
tau=2
alpha=0.2
beta=0.3
sigma=0.1
th=0 #trial
ee=0.1 #hindi ee
delta=0.1
epsilon=1
thcounter=0
#----------------------------------------
del_O= np.zeros((N,m)) #opinion vector
del_T=np.zeros((N,N,m))  #trust (sensitivity)
del_del_T=np.zeros((N,N,m))  #trust (sensitivity) last stored delta (for derivative of derivative)
del_C= np.zeros((N,m))   #credibility
del_I=np.zeros((N,m))    #influencing power

#store all values of these parameters as state progresses to track change
O_change=[[]for i in range(N)]
C_change=[[]for i in range(N)]
I_change=[[]for i in range(N)]
T_change=[[[]for j in range(N)]for i in range(N)]


pos_int=np.zeros((N,N))
tot_int=np.zeros((N,N))

Tf=0 #total time
int=np.nonzero(A) #int[0] is an array of person A interacting with person B in array int[1]
conn=len(int[0]) #total number of links in the network

for a in range(Tf):
     x=random.randint(0,conn-1) #randomly pick an edge
     i=int[0][x] #person1
     j=int[1][x] #person2
     no_top=random.randint(0,m) #how many topics they interact on
     s=random.sample(set(np.arange(m)), no_top) #pick no_top random topics
     for b in range(len(s)): #loop over the topics
          top=s[b]    #topic
          CO1=C[j][top]*O[j][top]-C[i][top]*O[i][top] #before update 
          ##########opinion########
          #print (i,'\n',O_change[i], 'old')
          #print ('\n',O_change[i], O[i],'new')
          th_val=T[i,j,top]*((C[j][top]*I[j][top])/(C[i][top]*I[i][top]+C[j][top]*I[j][top]))
          yo=(O[j][top]-O[i][top])*max(0,tanh(th_val-th))
          yoo=(1-(O[j][top]-np.dot(O[:,top]*I[:,top],A[i].T)))/np.count_nonzero(A[i])
          del_O[i][top]=yo*yoo
          O[i][top]+=del_O[i][top]   #change in opinion vector
          O_change[i].append(O[i])
          ##########credibility#########
          if th_val>=th:
               del_C[j][top]= epsilon*tanh(del_O[i][top])
               C[j][top]+= del_C[j][top]#j's credibility goes up if i's opinion changed above the threshold.
               pos_int[i][j]+=1
               tot_int[i][j]+=1
          else:
               #print("C'est la vie")
               thcounter+=1
               del_C[j][top]=delta
               C[j][top]=max(0,C[j][top]-del_C[j][top]) #j credibility goes down
               tot_int[i][j]+=1
          C_change[i].append(C[i])
          #########influencing_power########
          I[j][top]=max(0,I[j][top]+sigma*del_C[j][top]) #if j's credibility changed so does their influencing power
          I_change[i].append(I[i])
          ##########trust############
          CO2=C[j][top]*O[j][top]-C[i][top]*O[i][top] #after update 
          yo=alpha*(del_T[i][j][top]-del_del_T[i][j][top])
          if tot_int[i][j]!=0 and scipy.spatial.distance.cosine(O[i],O[j])!=0:
               yoyo=beta*((CO2-CO1)/(scipy.spatial.distance.cosine(O[i],O[j])))*pos_int[i,j]/tot_int[i][j]
          else:
               yoyo=0
          del_del_T[i,j,top]=del_T[i,j,top]
          del_T[i,j,top]=yo + yoyo
          T[i,j,top]=min(max(0,T[i,j,top]+del_T[i,j,top]),1)
          T_change[i][j].append(T[i,j])


###########plotting and clustering ##########################

O_change=np.array(O_change)
C_change=np.array(C_change)
I_change=np.array(I_change)
T_change=np.array(T_change)

#clustering all patients to see how many communities there are
temphist=[]
Z=np.zeros((N,N))
for i in range(N):
     for j in range(N):
          Z[i,j]=scipy.spatial.distance.cosine(O[i],O[j])+0.5
          temphist.append(Z[i,j])

'''plt.hist(temphist)
plt.show()
if m==2:
     plt.scatter(O[:,0],O[:,1]);
     plt.show()
'''

H=nx.Graph(Z)              
part = community.best_partition(H)
values = [part.get(node) for node in H.nodes()]
mod = community.modularity(part,H)

nocomm=len(np.unique(values))
print ('number of communities in the system after time ',Tf,' is: ',nocomm)
          
print ('And the population of each community is:', Counter(values))
print ('And communities are:', values)     
print(str(thcounter))

#plotting the parameters over all people

pal=igraph.RainbowPalette(n=N)
yo=[pal.get(i) for i in range(N)]


'''for i in range(N):
     s=copy.deepcopy(np.array(O_change[i]))
     if s.any():
          #if this person has at all has any change /interactions
          plot(s[:,2],color=yo[i]) #3rd topic credibility change
xlabel('Time',fontsize=18)
ylabel('Credibility',fontsize=18)'''

'''plt.subplot(2, 2, 2)
for i in range(N):
     plot(I[i][2],color=yo[i]) #3rd topic influencing power change
xlabel('Time',fontsize=18)
ylabel('Credibility',fontsize=18)


plt.subplot(2, 2, 3)
for i in range(N):
     plot(O[i][2],color=yo[i]) #3rd topic opinion change
xlabel('Time',fontsize=18)
ylabel('Credibility',fontsize=18)


plt.subplot(2, 2, 4)
for i in range(N):
     plot(O[i][3],color=yo[i]) #4th topic opinion change
xlabel('Time',fontsize=18)
ylabel('Credibility',fontsize=18)'''

show()

