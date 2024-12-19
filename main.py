import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
num_years = np.loadtxt('num_years.txt')
depths_uniform = np.loadtxt('dep.txt')
print(depths_uniform,depths_uniform.shape)
NY = num_years.shape
print(num_years,num_years.shape)
depths_uniform = depths_uniform.reshape(34,36)
print(depths_uniform,depths_uniform.shape)

distances_uniform =np.loadtxt('dist_unif.txt')
ND = distances_uniform.shape
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X,Y = np.meshgrid(distances_uniform,num_years)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, depths_uniform, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
# ax.set_ylabel('YEAR')
# ax.set_xlabel('DISTANCE')
# plt.show(block=True)
# plt.savefig('dep.png')

future_points = 3
from predict_auto import predict
from predict import make_series
dep_predicted = []#np.zeros((depths_uniform.T.shape[0],test.shape[0]+future_points))
for i,d in enumerate(depths_uniform.T):
    s = make_series(d,num_years,str(distances_uniform[i]))

    # return EXTENDED PREDICTIONS, I.E. PREDICTIONS UNITED WITH TRAIN SET
    predictions,mape,num_years_prolonged = predict(s,future_points,str(distances_uniform[i]),num_years)

    dep_predicted.append(predictions)
qq = 0
dep_predicted = np.array(dep_predicted)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X,Y = np.meshgrid(distances_uniform,num_years_prolonged)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, dep_predicted.T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf)
ax.set_ylabel('YEAR')
ax.set_xlabel('DISTANCE')
plt.title('Coast profile evolution prediction with ARIMA')
plt.show(block=True)
plt.savefig('dep_ARIMA.png')

