import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_spiral_data(n_points, noise=1.0):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    
    # Spiral 1
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    
    # Spiral 2 - Offset by 2*pi/3
    d2x = -np.cos(n + 2*np.pi/3) * n + np.random.rand(n_points, 1) * noise
    d2y = np.sin(n + 2*np.pi/3) * n + np.random.rand(n_points, 1) * noise
    
    # Spiral 3 - Offset by 4*pi/3
    d3x = -np.cos(n + 4*np.pi/3) * n + np.random.rand(n_points, 1) * noise
    d3y = np.sin(n + 4*np.pi/3) * n + np.random.rand(n_points, 1) * noise
    
    # Combine the spirals and assign labels
    data = np.vstack((np.hstack((d1x, d1y, np.ones((n_points, 1)))), 
                      np.hstack((d2x, d2y, 2*np.ones((n_points, 1)))), 
                      np.hstack((d3x, d3y, 3*np.ones((n_points, 1))))))
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['X', 'y', 'spiral'])
    
    return df

if __name__ == "__main__":
    # Generate the dataset
    df = generate_spiral_data(100)

    # Visualize the dataset with different colors for each spiral
    colors = {1: 'red', 2: 'blue', 3: 'green'}
    plt.scatter(df['X'], df['y'], c=df['spiral'].apply(lambda x: colors[x]))
    plt.title('Spiral Dataset')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
