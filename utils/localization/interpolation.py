from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt

class interpolation():
    
    def interpolation_1d(self, arr_x, arr_y,min_dist=0.1):

        # Create interpolation function
        f = interp1d(arr_x, arr_y, kind='linear')  # or 'cubic', 'quadratic', etc.

        # Interpolated values
        x_new = np.linspace(0, 4, 50)
        y_new = f(x_new)
        
        return (x_new, y_new)
    
    def interpolation_np(self, arr_x, arr_y,min_dist=0.1):
        x_new = 1.5
        y_new = np.interp(x_new, arr_x, arr_y)
        print(y_new)  # Linear interpolation at x=1.5
        
    def interpolation_2d(self):

        # Example points
        points = np.array([[0, 0], [0,1], [1,0], [1, 1]])
        values = np.array([0, 1, 1, 0])

        print(points.T)
        pints = np.array([[0,0],[1,0],[2,0],[3,0]])
        print(pints.T)

        # Grid to interpolate onto
        grid_x, grid_y = np.mgrid[0:1:10j, 0:1:10j]

        # Interpolation
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        
        plt.imshow(grid_z.T, extent=(0, 1, 0, 1), origin='lower')
        
        # Plot the result
        plt.imshow(grid_z.T, extent=(0, 1, 0, 1), origin='lower')
        plt.scatter(points[:,0], points[:,1], c=values, edgecolor='k')
        plt.colorbar(label='Interpolated value')
        plt.title('2D Interpolation with griddata')
        plt.show()
        
def main():
    # Sample data
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])
    
    interp = interpolation()
    # x_new, y_new = interp.interpolation_1d(x,y)
    # interp.interpolation_np(x,y)
    interp.interpolation_2d()

    # Plotting
    # plt.plot(x, y, 'o', label='Original Data')
    # plt.plot(x_new, y_new, '-', label='Interpolated')
    # plt.legend()
    # plt.show()
    
    
if __name__ == '__main__':
    main()