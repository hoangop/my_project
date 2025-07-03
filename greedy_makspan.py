import heapq
def greedy_makespan_min(times, m):
    # times is a list of n jobs.
    assert len(times) >= 1
    assert all(elt >= 0 for elt in times)
    assert m >= 2
    n = len(times)
    # please do not reorder the jobs in times or else tests will fail.
    # you can implement a priority queue if you would like.
    # use https://docs.python.org/3/library/heapq.html heapq data structure 
    # Return a tuple of two things: 
    #    - Assignment list of n numbers from 0 to m-1
    #    - The makespan of your assignment
    # your code here
    
    # Initialize the assignment list. This list will store the processor index
    assign = [0] * n

    # Initialize for each processor.
    processor_loads = [0] * m
    
    # Initialize a min-heap.
    min_heap = []
    for j in range(m):
        heapq.heappush(min_heap, (0, j)) 

    # Iterate through each job in the order they are given in the 'times' list.
    for i in range(n):
        job_duration = times[i]
        current_least_load, processor_idx = heapq.heappop(min_heap)
        assign[i] = processor_idx
        new_load = current_least_load + job_duration
        processor_loads[processor_idx] = new_load
        heapq.heappush(min_heap, (new_load, processor_idx))

    # After all jobs have been assigned, the makespan is the maximum load
    makespan = max(processor_loads)

    # Return the generated assignment list and its corresponding makespan.
    return assign, makespan 

def compute_makespan(times, m, assign):
    # times is an array of job times of size n
    # m is the number of processors
    # assign is an array of size n whose entries are between 0 to m-1 
    # indicating the processor number for
    # the corresponding job.
    # Return: makespan of the assignment
    # my code here

    
    # 1. Initialize for each processor.
    processor_times = [0] * m

    # 2. Iterate through each job and add its time to its assigned processor.
    n = len(times)
    
    for i in range(n):
        job_duration = times[i]

        assigned_processor_idx = assign[i]

        if 0 <= assigned_processor_idx < m:
            processor_times[assigned_processor_idx] += job_duration
        else:
            raise ValueError(
                f"Invalid processor index {assigned_processor_idx} for job {i}. "
                f"Processor index must be between 0 and {m-1}."
            )

    # 3. Calculate the makespan.
    if not processor_times:
        return 0 # No processors means no makespan, or 0 if no work can be done.
        
    makespan = max(processor_times)

    return makespan    

## BEGIN TESTS
def do_test(times, m, expected):
    (a, makespan) = greedy_makespan_min(times,m )
    print('\t Assignment returned: ', a)
    print('\t Claimed makespan: ', makespan)
    assert compute_makespan(times, m, a) == makespan, 'Assignment returned is not consistent with the reported makespan'
    assert makespan == expected, f'Expected makespan should be {expected}, your core returned {makespan}'
    print('Passed')
print('Test 1:')
times = [2, 2, 2, 2, 2, 2, 2, 2, 3] 
m = 3
expected = 7
do_test(times, m, expected)

print('Test 2:')
times = [1]*20 + [5]
m = 5
expected =9
do_test(times, m, expected)

print('Test 3:')
times = [1]*40 + [2]
m = 20
expected = 4
do_test(times, m, expected)
print('All tests passed: 15 points!')
## END TESTS