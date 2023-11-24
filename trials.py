import matplotlib.pyplot as plt
import numpy as np

# Example lists with n arrays, where len(A[i]) == len(B[i]) for all i
n = 10
A = [np.linspace(0, 10, 100 + i * 10) for i in range(n)]  # Example A arrays
B = [np.sin(a) for a in A]  # Example B arrays (sin function of A)

# Create 10 individual plots
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
fig.tight_layout()

for i in range(n):
    row, col = divmod(i, 5)
    ax = axs[row, col]

    # Plot B[i] as a function of A[i]
    ax.plot(A[i], B[i])
    ax.set_title(f'Plot {i + 1}')
    ax.set_xlabel('A[i]')
    ax.set_ylabel('B[i]')

plt.show()
