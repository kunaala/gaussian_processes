import numpy as np
import matplotlib.pyplot as plt

with open("plot.csv",'r') as f:
    lines = f.readlines()

dim = np.fromstring(lines[0], dtype=float, sep = ",")
x = np.fromstring(lines[1], dtype=float, sep = ",")
y = np.fromstring(lines[2], dtype=float, sep = ",")
x_t = np.fromstring(lines[3], dtype=float, sep = ",")
# y_t = np.fromstring(lines[4], dtype=float, sep = ",")
mu_t = np.fromstring(lines[4], dtype=float, sep = ",")
s = np.fromstring(lines[5], dtype=float, sep = ",")

plt.figure(figsize=(16,8))


plt.plot(x, y,'k+', ms=8)
plt.plot(x_t, mu_t,'b-' , ms = 18)
# plt.plot(xp,yp,'r+', ms=14)
plt.gca().fill_between(x_t, mu_t-2*s, mu_t+2*s, color="#dddddd")

plt.savefig("predictive.png", bbox_inches='tight', dpi=300)
plt.title('Mean predictions plus 2 st.deviations')
plt.show()

