# Pattern Recognition ---- Assignment#(2-2)
# Salar Nouri
#-------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
import math

def normpdf(x, mu, var):
    pi = 3.1415926
    num = math.exp(-(float(x)-float(mu))**2/(2*var))
    return num/((2*pi*var)**.5)

def calc_p_theta(i):
	res= 0;
	for j in range(0,J):
		res= res+ normpdf(samples[i],mus[j],vars[j])*pis[j]
	return res

def update_pis_cdt():
	for i in range(0,np.size(samples)):
		downside= calc_p_theta(i)
		for j in range(0,J):
			pis_cdt[j][i]= (normpdf(samples[i],mus[j],vars[j])*pis[j])/downside

def update_pis():
	for j in range(0,J):
		pis[j]= 0
		for i in range(0, np.size(samples)):
			pis[j]= pis[j] + ((1/np.size(samples))*pis_cdt[j][i])

def update_mus():
	for j in range(0,J):
		downside= np.sum(pis_cdt[j])
		upside= 0
		for i in range(0,np.size(samples)):
			upside= upside + pis_cdt[j][i]*samples[i]
		mus[j]= upside/downside

def update_vars():
	for j in range(0,J):
		downside= np.sum(pis_cdt[j])
		upside= 0
		for i in range(0,np.size(samples)):
			upside= upside + pis_cdt[j][i]*((samples[i]-mus[j])**2)
		vars[j]= upside/downside

def mdm_step():
	update_pis_cdt()
	update_pis()
	update_mus()
	update_vars()

def calc_Q():
	res= 0
	for i in range(0,np.size(samples)):
		for j in range(0,J):
			res = res + (pis_cdt[j][i]*( -0.5*np.log(vars[j]) +
			 (-0.5/vars[j])*((samples[i]-mus[j])**2) + np.log(pis[j])))
			#print (res)
	return res


J= 4
TRSH= 1e-7
NUM_OF_SAMPLES= 750
NUM_OF_DISTS= 4
Q= -100000

mu1, sigma1 = 1, 0.1
mu2, sigma2 = 1.5, math.sqrt(0.1)
mu3, sigma3 = 2, 0.2
mu4, sigma4 = 2.6, math.sqrt(0.5)
s1 = np.random.normal(mu1, sigma1, NUM_OF_SAMPLES);
s2 = np.random.normal(mu2, sigma2, NUM_OF_SAMPLES);
s3 = np.random.normal(mu3, sigma3, NUM_OF_SAMPLES);
s4 = np.random.normal(mu4, sigma4, NUM_OF_SAMPLES);

samples=[]
for i in range(0,NUM_OF_SAMPLES,2):
	samples.extend([s1[i],s4[i],s4[i],s4[i]])
	samples.extend([s4[i+1],s1[i+1],s1[i+1],s1[i+1]])

mus= np.random.uniform(0,4,J)
vars= np.random.uniform(0,0.5,J)
pis= np.random.uniform(0,1,J)
pis= pis/np.sum(pis)
pis_cdt= np.zeros(shape=(J, np.size(samples)))

new_Q= calc_Q()
qlist=[];
i=0

while (i<40):
	print (new_Q)
	Q= new_Q
	mdm_step()
	new_Q= calc_Q()
	qlist.append(new_Q)
	print(pis, "\n", mus, "\n", vars,"\n")
	i=i+1

sg1= np.random.normal(mus[0],math.sqrt(vars[0]), int(np.floor(pis[0]*NUM_OF_SAMPLES*4)))
sg2= np.random.normal(mus[1],math.sqrt(vars[1]), int(np.floor(pis[1]*NUM_OF_SAMPLES*4)))
sg3= np.random.normal(mus[2],math.sqrt(vars[2]), int(np.floor(pis[2]*NUM_OF_SAMPLES*4)))
sg4= np.random.normal(mus[3],math.sqrt(vars[3]), int(np.floor(pis[3]*NUM_OF_SAMPLES*4)))

samples_est= np.concatenate((sg1,sg2,sg3,sg4))

count, bins, ignored = plt.hist(samples, 30, density=True)

count, bins, ignored = plt.hist(samples_est, 30, alpha=0.5, density=True)
plt.legend(["training set","generated"])
plt.show()

plt.plot(qlist)
plt.show()
