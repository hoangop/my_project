from math import sqrt

from matplotlib import pyplot as plt 


def euclidean_distance(a, b):
    (xa, ya) = a
    (xb, yb) = b
    return sqrt( (xb - xa)**2 + (yb - ya)**2)

def calculate_R(coords, center_indices):
    n = len(coords)
    assert all( 0 <= j < n for j in center_indices)
    rj_values = [ min([euclidean_distance(xj, coords[j]) for j in center_indices]) for xj in coords]
    return max(rj_values)

def plot_coords(coords, center_indices):
    R = calculate_R(coords, center_indices)
    coords_x = [x for (x,y) in coords]
    coords_y = [y for (x, y) in coords]
    centers_x = [coords_x[j] for j in center_indices]
    centers_y = [coords_y[j] for j in center_indices]
    figure, axes = plt.subplots()
    axes.axis('equal')
    for k in center_indices:
        c = plt.Circle(coords[k], R, fill=True, alpha=0.5, facecolor='lightblue', clip_on=False, edgecolor='black', linewidth=1, linestyle='dashed')
        axes.add_artist(c)
    plt.scatter(coords_x, coords_y, s=30, marker='x' )
    plt.scatter(centers_x, centers_y, s=50, marker='o')
    plt.show()

def euclidean_distance(a, b):
    (xa, ya) = a
    (xb, yb) = b
    return sqrt( (xb - xa)**2 + (yb - ya)**2)


# Function find_farthest_point_from_current_centers
# returns a pair (j, rj) where 
# - 0 <= j < len(coords) is the index of the farthest point P_j
# - rj is the distance of the point P_j from its nearest center
def find_farthest_point_from_current_centers(coords, center_indices):
    n = len(coords)
    assert all( 0 <= j < n for j in center_indices)
    # For each point xi, find the minimum distance to any of the current centers.
    # Then, find the maximum among all these minimum distances.
    rj_values = [ (min([euclidean_distance(xi, coords[j]) for j in center_indices]), i) for (i, xi) in enumerate(coords)]
    (rj, j) = max(rj_values)
    return (j, rj)

## Implement a function greedy_k_centers that given a list of coordinates `coords`, returns center_list, R
##   - centers_list is a list of indices [j1,..., jk]. Note that coords[j1], ..., coords[jk] will yield coordinates of the actual centetr.
##   - R is the radius resulting from the choice of the k centers
## Please use the implementation of find_farthest_point_from_current_centers above.
def greedy_k_centers(coords, k, debug=True): ## Please print messages from this function only if debug flag is True
    # Start with the very first point as the initial center.
    centers = [0] 
    if debug:
        print(f'Tâm khởi tạo (chỉ số 0): {coords[0]}')
    # your code here        
    # We already have one center, so we need to find k-1 more.
    for i in range(k - 1):
        (farthest_point_index, distance) = find_farthest_point_from_current_centers(coords, centers)
        
        centers.append(farthest_point_index)
        
        if debug:
            print(f'Vòng lặp {i+2}/{k}: Thêm tâm mới (chỉ số {farthest_point_index}): {coords[farthest_point_index]}')

    # After finding all k centers, calculate the final radius R.
    # R is the maximum distance from any point to its nearest center in the final set.
    # This is exactly what the helper function returns as its second value.
    (_, R) = find_farthest_point_from_current_centers(coords, centers)
    
    if debug:
        print(f'---')
        print(f'Đã chọn xong {k} tâm (chỉ số): {centers}')
        print(f'Bán kính cuối cùng R: {R:.4f}')
        
    return centers, R

#test
from random import uniform
## Generate 1000 points
n = 1000
k = 12
coords = [(uniform(-2,-1), uniform(-2,2)) for i in range(n//4)] + [(uniform(-1,1), uniform(-1,1)) for i in range(n//4)] +  [(uniform(1,2), uniform(-2,0)) for i in range(n//4)] +  [(uniform(1,2), uniform(0,2)) for i in range(n//4)] 
            
(center_indices, R) = greedy_k_centers(coords, k, debug=False)
plot_coords(coords, center_indices)
def calculate_R(coords, center_indices):
    n = len(coords)
    assert all( 0 <= j < n for j in center_indices)
    rj_values = [ min([euclidean_distance(xj, coords[j]) for j in center_indices]) for xj in coords]
    return max(rj_values)

assert len(center_indices) == k
assert abs(R - calculate_R(coords, center_indices)) <= 1E-06, f'The returned value of R={R} from your function does not match with my computation. Something is wrong in your calculations'
print('Test Passed (5 points)')