'''
Takes in a path to a given log file and plots the total, generator, and discriminator loss 
'''

import sys
import matplotlib.pyplot as plt

if (len(sys.argv) != 2):
	print 'Please provide only one argument, the path to the file'
	exit()

f = open(sys.argv[1], 'r')

epochs = []
total_loss = []
generator_loss = []
discriminator_loss = [] 

graphingValues = [epochs, total_loss, generator_loss, discriminator_loss]

for line in f:
	if 'Epoch' in line: 
		holder = line.split('||')
		for i in xrange(len(holder)): 
			valueHolder = holder[i].split(' ')
			if i == len(holder) - 1:
				graphingValues[i].append((valueHolder[len(valueHolder) - 1]).replace('\n', ''))
			else: 
				graphingValues[i].append(valueHolder[len(valueHolder) - 2])

print epochs

# plot the errors 
plt.plot(epochs, total_loss, label = 'Total loss')
plt.plot(epochs, generator_loss, label = 'Generator loss')
plt.plot(epochs, discriminator_loss, label = 'Discriminator loss')
plt.title('Loss Plots \n')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
#plt.ylim(ymin = 0, ymax = 120)
plt.legend()
plt.show()

f.close()