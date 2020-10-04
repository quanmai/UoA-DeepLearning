import matplotlib.pyplot as plt
import numpy as np

def f(x):
    t= 61+np.sqrt(10777)
    a = 84/t
    b = (-(394-8*np.sqrt(10777))/3) / t;
    return a*x + b
   
x = np.linspace(0, 10, 1000)
fig1 = plt.gcf()
plt.axes().set_aspect('equal')
plt.xlim(0,10)
plt.ylim(0,10)
c1 = np.array([[2,1],[2,2],[2,3]])
c2 = np.array([[4,3],[5,3],[6,4]])
c3 = [3.5, 8/3]
plt.plot(x, f1(x))
plt.scatter(c1[:,0],c1[:,1],c='r')
plt.scatter(c2[:,0],c2[:,1],c='y')
plt.scatter(c3[0],c3[1],c='g')
plt.legend(["PCA", "c1", "c2", "sample mean"], loc ="upper right");
plt.show();
plt.draw();
fig1.savefig('PCA.png')
