
import numpy as np
import matplotlib.pyplot as plt

a=np.arange(0, 0.55, 0.05)
c=(1-2.*a)*np.log2(1-2.*a)+2.*a*np.log2(a)+2
c[0]=2
c[len(c)-1]=1
plt.plot(a, c, marker='o')
plt.xlabel(r'$\alpha$')
plt.ylabel('Capacity')
plt.show()

print(c)


