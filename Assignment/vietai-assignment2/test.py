import numpy as np

print(np.maximum(1, -5))

a = np.array([1, 2, 3, -5, 6.6])

# a[a <= 0] = 0
# a[a > 0] = 1
# print(a)


grad =1*(a > 0)

for   i in range(100):
    print(i);
print(grad)
