from math import sqrt

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
        
    # We already have one center, so we need to find k-1 more.
    for i in range(k - 1):
        # Find the point that is farthest from any of the existing centers.
        (farthest_point_index, distance) = find_farthest_point_from_current_centers(coords, centers)
        
        # Add this farthest point to our list of centers.
        centers.append(farthest_point_index)
        
        if debug:
            # Iteration i+2 corresponds to finding the j-th center where j = i+2
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