// # Complete C++ STL Reference Guide with Use Cases

// ## 1. SEQUENCE CONTAINERS

// ### **vector** - Dynamic Array
// ```cpp
#include <bits/stdc++.h>
using namespace std;
#include <vector>

int n, val, v, v1;


// DECLARATION & INITIALIZATION
vector<int> v;                          // Empty vector
vector<int> v(n);                       // n elements, default initialized
vector<int> v(n, val);                  // n elements with value val
vector<int> v = {1, 2, 3};             // Initializer list
vector<int> v2(v1);                     // Copy constructor

// OPERATIONS                           Time        Space       Use Case
v.push_back(x);                        // O(1)*      O(1)        Add element to end (growing list)
v.pop_back();                          // O(1)       O(1)        Remove last element (stack operations)
v.insert(it, x);                       // O(n)       O(1)        Insert at specific position
v.insert(it, n, x);                    // O(n)       O(1)        Insert n copies of x
v.erase(it);                           // O(n)       O(1)        Remove element at position
v.erase(it1, it2);                     // O(n)       O(1)        Remove range of elements
v.clear();                             // O(n)       O(1)        Remove all elements
v.resize(n);                           // O(n)       O(1)        Change size (truncate/expand)
v.reserve(n);                          // O(n)       O(1)        Pre-allocate memory (optimization)
v.shrink_to_fit();                     // O(n)       O(1)        Release unused memory
v.assign(n, val);                      // O(n)       O(1)        Replace contents with n copies
v.emplace_back(args);                  // O(1)*      O(1)        Construct element in-place
v.emplace(it, args);                   // O(n)       O(1)        Construct at position

// ACCESS                               Time        Space       Use Case
v[i];                                  // O(1)       O(1)        Fast unchecked access
v.at(i);                               // O(1)       O(1)        Bounds-checked access (safe)
v.front();                             // O(1)       O(1)        First element access
v.back();                              // O(1)       O(1)        Last element access
v.data();                              // O(1)       O(1)        Get raw array pointer (C API)

// CAPACITY                             Time        Space       Use Case
v.size();                              // O(1)       O(1)        Current number of elements
v.empty();                             // O(1)       O(1)        Check if container is empty
v.capacity();                          // O(1)       O(1)        Allocated storage capacity
v.max_size();                          // O(1)       O(1)        Maximum possible size

// COMMON USE CASES
// 1. Dynamic array when size changes
vector<int> scores;
scores.push_back(95);

// 2. Stack implementation
vector<int> stack;
stack.push_back(10);  // push
int top = stack.back(); // top
stack.pop_back();      // pop

// 3. Matrix representation
vector<vector<int>> matrix(rows, vector<int>(cols, 0));

// 4. Buffer for I/O operations
vector<char> buffer(1024);

// 5. Dynamic programming table
vector<int> dp(n + 1, 0);
// ```

### **deque** - Double-Ended Queue ###
// ```cpp
#include <deque>

// OPERATIONS                           Time        Space       Use Case
dq.push_back(x);                       // O(1)       O(1)        Add to rear (queue operations)
dq.push_front(x);                      // O(1)       O(1)        Add to front (deque specific)
dq.pop_back();                         // O(1)       O(1)        Remove from rear
dq.pop_front();                        // O(1)       O(1)        Remove from front (queue operations)
dq.insert(it, x);                      // O(n)       O(1)        Insert at position
dq.erase(it);                          // O(n)       O(1)        Remove at position

// ACCESS
dq[i];                                 // O(1)       O(1)        Random access
dq.at(i);                              // O(1)       O(1)        Bounds-checked access
dq.front();                            // O(1)       O(1)        First element
dq.back();                             // O(1)       O(1)        Last element

// COMMON USE CASES
// 1. Sliding window problems
deque<int> window;
for (int num : nums) {
    while (!window.empty() && window.front() < num - k)
        window.pop_front();
    window.push_back(num);
}

// 2. Palindrome checker
deque<char> dq(s.begin(), s.end());
while (dq.size() > 1) {
    if (dq.front() != dq.back()) return false;
    dq.pop_front();
    dq.pop_back();
}

// 3. BFS with level-order traversal
deque<TreeNode*> queue;
queue.push_back(root);

// 4. Implement both stack and queue
deque<int> ds;
ds.push_back(1);   // Can act as stack
ds.push_front(2);  // Can act as queue
```

### **list** - Doubly Linked List
```cpp
#include <list>

// OPERATIONS                           Time        Space       Use Case
lst.push_back(x);                      // O(1)       O(1)        Add to end
lst.push_front(x);                     // O(1)       O(1)        Add to beginning
lst.pop_back();                        // O(1)       O(1)        Remove from end
lst.pop_front();                       // O(1)       O(1)        Remove from beginning
lst.insert(it, x);                     // O(1)       O(1)        Insert at iterator position
lst.erase(it);                         // O(1)       O(1)        Remove at iterator position
lst.remove(val);                       // O(n)       O(1)        Remove all occurrences
lst.remove_if(pred);                   // O(n)       O(1)        Remove if condition met
lst.unique();                          // O(n)       O(1)        Remove consecutive duplicates
lst.sort();                            // O(nlogn)   O(1)        Sort list
lst.reverse();                         // O(n)       O(1)        Reverse list
lst.merge(lst2);                       // O(n)       O(1)        Merge sorted lists
lst.splice(it, lst2);                  // O(1)       O(1)        Transfer elements

// COMMON USE CASES
// 1. LRU Cache implementation
list<pair<int, int>> cache;
unordered_map<int, list<pair<int,int>>::iterator> map;

// 2. Undo/Redo functionality
list<string> history;
auto current = history.begin();

// 3. Music playlist (frequent insertions/deletions)
list<Song> playlist;
playlist.push_back(song1);
playlist.remove(song2);

// 4. Process scheduling
list<Process> processes;
processes.splice(processes.end(), processes, it); // Move to end
```

### **forward_list** - Singly Linked List
```cpp
#include <forward_list>

// OPERATIONS                           Time        Space       Use Case
fl.push_front(x);                      // O(1)       O(1)        Add to front only
fl.pop_front();                        // O(1)       O(1)        Remove from front
fl.insert_after(it, x);                // O(1)       O(1)        Insert after position
fl.erase_after(it);                    // O(1)       O(1)        Erase after position
fl.remove(val);                        // O(n)       O(1)        Remove all occurrences
fl.unique();                           // O(n)       O(1)        Remove consecutive duplicates
fl.sort();                             // O(nlogn)   O(1)        Sort list
fl.reverse();                          // O(n)       O(1)        Reverse list

// COMMON USE CASES
// 1. Memory-constrained environments (less overhead than list)
forward_list<int> fl = {1, 2, 3};

// 2. Graph adjacency list
vector<forward_list<int>> graph(n);

// 3. Simple stack implementation
forward_list<int> stack;
stack.push_front(10);
stack.pop_front();
```

// ### **array** - Fixed-Size Array ### 

#include <array>

// OPERATIONS                           Time        Space       Use Case
arr[i];                                // O(1)       O(1)        Index access
arr.at(i);                             // O(1)       O(1)        Bounds-checked access
arr.front();                           // O(1)       O(1)        First element
arr.back();                            // O(1)       O(1)        Last element
arr.fill(val);                         // O(n)       O(1)        Fill with value
arr.swap(arr2);                        // O(n)       O(1)        Swap contents

// COMMON USE CASES
// 1. Fixed-size buffers
array<char, 256> buffer{};

// 2. Matrix with compile-time dimensions
array<array<int, 3>, 3> matrix{};

// 3. Lookup tables
array<int, 128> ascii_values{};

// 4. Coordinate storage
array<int, 3> point3D = {x, y, z};
// ```

// ## 2. CONTAINER ADAPTERS

// ### **stack** - LIFO Container
// ```cpp
#include <stack>

// OPERATIONS                           Time        Space       Use Case
st.push(x);                            // O(1)       O(1)        Add to top
st.pop();                              // O(1)       O(1)        Remove from top
st.top();                              // O(1)       O(1)        Access top element
st.empty();                            // O(1)       O(1)        Check if empty
st.size();                             // O(1)       O(1)        Get size

// COMMON USE CASES
// 1. Expression evaluation
stack<int> operands;
stack<char> operators;

// 2. Parentheses matching
stack<char> st;
for (char c : expression) {
    if (c == '(') st.push(c);
    else if (c == ')') {
        if (st.empty()) return false;
        st.pop();
    }
}

// 3. Function call stack simulation
stack<CallFrame> callStack;

// 4. Undo operations
stack<Command> undoStack;

// 5. DFS traversal
stack<Node*> st;
st.push(root);
while (!st.empty()) {
    Node* curr = st.top(); st.pop();
    // process
}

// 6. Next greater element
stack<int> st;
for (int i = n-1; i >= 0; i--) {
    while (!st.empty() && st.top() <= arr[i])
        st.pop();
    result[i] = st.empty() ? -1 : st.top();
    st.push(arr[i]);
}
// ```

// ### **queue** - FIFO Container
// ```cpp
#include <queue>

// OPERATIONS                           Time        Space       Use Case
q.push(x);                             // O(1)       O(1)        Add to rear
q.pop();                               // O(1)       O(1)        Remove from front
q.front();                             // O(1)       O(1)        Access front element
q.back();                              // O(1)       O(1)        Access rear element
q.empty();                             // O(1)       O(1)        Check if empty
q.size();                              // O(1)       O(1)        Get size

// COMMON USE CASES
// 1. BFS traversal
queue<Node*> q;
q.push(root);
while (!q.empty()) {
    Node* curr = q.front(); q.pop();
    for (Node* child : curr->children)
        q.push(child);
}

// 2. Task scheduling
queue<Task> taskQueue;

// 3. Level-order traversal
queue<TreeNode*> q;

// 4. Producer-consumer problem
queue<Data> buffer;

// 5. Print job spooling
queue<PrintJob> printQueue;


// ### **priority_queue** - Heap
#include <queue>

// DECLARATION
priority_queue<int> pq;                              // Max heap
priority_queue<int, vector<int>, greater<int>> pq;  // Min heap

// OPERATIONS                           Time        Space       Use Case
pq.push(x);                            // O(logn)    O(1)        Insert element
pq.pop();                              // O(logn)    O(1)        Remove top element
pq.top();                              // O(1)       O(1)        Access top element
pq.empty();                            // O(1)       O(1)        Check if empty
pq.size();                             // O(1)       O(1)        Get size

// COMMON USE CASES
// 1. Dijkstra's algorithm
priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
pq.push({0, source});

// 2. K largest/smallest elements
priority_queue<int, vector<int>, greater<int>> minHeap;
for (int num : nums) {
    minHeap.push(num);
    if (minHeap.size() > k) minHeap.pop();
}

// 3. Merge K sorted lists
auto comp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
priority_queue<ListNode*, vector<ListNode*>, decltype(comp)> pq(comp);

// 4. Task scheduling with priorities
struct Task {
    int priority;
    string name;
    bool operator<(const Task& other) const {
        return priority < other.priority; // Higher priority first
    }
};
priority_queue<Task> tasks;

// 5. Median finder
priority_queue<int> maxHeap;  // Lower half
priority_queue<int, vector<int>, greater<int>> minHeap;  // Upper half

// 6. Huffman coding
priority_queue<Node*, vector<Node*>, CompareNodes> pq;
```

// ## 3. ASSOCIATIVE CONTAINERS

// ### **set** - Ordered Unique Elements
// ```cpp
#include <set>

// OPERATIONS                           Time        Space       Use Case
s.insert(x);                           // O(logn)    O(1)        Add element
s.erase(x);                            // O(logn)    O(1)        Remove element
s.erase(it);                           // O(1)*      O(1)        Remove by iterator
s.find(x);                             // O(logn)    O(1)        Search element
s.count(x);                            // O(logn)    O(1)        Check existence (0 or 1)
s.lower_bound(x);                      // O(logn)    O(1)        First element >= x
s.upper_bound(x);                      // O(logn)    O(1)        First element > x
s.equal_range(x);                      // O(logn)    O(1)        Range of equal elements

// COMMON USE CASES
// 1. Remove duplicates while maintaining order
set<int> s(vec.begin(), vec.end());
vec.assign(s.begin(), s.end());

// 2. Range queries
set<int> s = {1, 3, 5, 7, 9};
auto it = s.lower_bound(4);  // Points to 5

// 3. Running median
set<pair<int, int>> s;  // {value, index} for handling duplicates

// 4. Event scheduling (no conflicts)
set<pair<int, int>> intervals;  // {start, end}

// 5. Ordered statistics
set<int> s;
auto kth = next(s.begin(), k-1);  // k-th smallest

// 6. Finding closest element
auto it = s.lower_bound(target);
int closest = *it;
if (it != s.begin()) {
    --it;
    if (abs(*it - target) < abs(closest - target))
        closest = *it;
}
```

### **multiset** - Ordered Duplicates Allowed
```cpp
#include <set>

// OPERATIONS                           Time        Space       Use Case
ms.insert(x);                          // O(logn)    O(1)        Add element
ms.erase(x);                           // O(logn)    O(1)        Remove ALL occurrences
ms.erase(it);                          // O(1)*      O(1)        Remove ONE occurrence
ms.count(x);                           // O(logn+k)  O(1)        Count occurrences
ms.find(x);                            // O(logn)    O(1)        Find first occurrence
ms.equal_range(x);                     // O(logn)    O(1)        Range of equal elements

// COMMON USE CASES
// 1. Frequency tracking with order
multiset<int> ms = {1, 2, 2, 3, 3, 3};
cout << ms.count(3);  // Output: 3

// 2. Sliding window median
multiset<int> window;
window.insert(nums[i]);
window.erase(window.find(nums[i-k]));  // Remove specific occurrence

// 3. Top K frequent elements (ordered)
multiset<pair<int, int>> ms;  // {frequency, element}

// 4. Range sum queries
multiset<int> ms;
auto range = ms.equal_range(x);
int sum = 0;
for (auto it = range.first; it != range.second; ++it)
    sum += *it;

// 5. Meeting rooms (overlapping allowed)
multiset<pair<int, int>> meetings;
```

### **map** - Ordered Key-Value Pairs
```cpp
#include <map>

// OPERATIONS                           Time        Space       Use Case
mp[key] = val;                         // O(logn)    O(1)        Insert/Update
mp.insert({key, val});                 // O(logn)    O(1)        Insert if not exists
mp.insert_or_assign(key, val);         // O(logn)    O(1)        Insert or update [C++17]
mp.erase(key);                         // O(logn)    O(1)        Remove by key
mp.find(key);                          // O(logn)    O(1)        Search key
mp.count(key);                         // O(logn)    O(1)        Check existence
mp.at(key);                            // O(logn)    O(1)        Access with exception
mp.lower_bound(key);                   // O(logn)    O(1)        First key >= key
mp.upper_bound(key);                   // O(logn)    O(1)        First key > key

// COMMON USE CASES
// 1. Frequency counter
map<char, int> freq;
for (char c : str) freq[c]++;

// 2. Sorted dictionary
map<string, string> dictionary;

// 3. Range sum with keys
map<int, int> prefixSum;
auto it = prefixSum.upper_bound(right);
auto it2 = prefixSum.lower_bound(left);

// 4. Interval mapping
map<int, int> intervals;  // start -> end

// 5. Leaderboard/Ranking
map<int, string, greater<int>> leaderboard;  // score -> name

// 6. Time-based key-value store
map<int, string> timeMap;  // timestamp -> value
auto it = timeMap.upper_bound(timestamp);
if (it != timeMap.begin()) --it;

// 7. Coordinate compression
map<int, int> compress;
int idx = 0;
for (int val : sorted_vals)
    compress[val] = idx++;
```

### **multimap** - Multiple Values per Key
```cpp
#include <map>

// OPERATIONS                           Time        Space       Use Case
mm.insert({key, val});                 // O(logn)    O(1)        Insert key-value
mm.erase(key);                         // O(logn)    O(1)        Remove ALL with key
mm.count(key);                         // O(logn+k)  O(1)        Count values for key
mm.equal_range(key);                   // O(logn)    O(1)        Get all values for key

// COMMON USE CASES
// 1. Graph adjacency list (ordered)
multimap<int, int> graph;
graph.insert({1, 2});
graph.insert({1, 3});

// 2. Index mapping
multimap<string, int> index;  // word -> positions

// 3. Event scheduling (multiple events at same time)
multimap<int, string> schedule;  // time -> event

// 4. Group by key
multimap<string, Person> groups;  // department -> employees
```

## 4. UNORDERED CONTAINERS

### **unordered_set** - Hash Set
```cpp
#include <unordered_set>

// OPERATIONS                           Time Avg    Time Worst  Use Case
us.insert(x);                          // O(1)       O(n)        Add element
us.erase(x);                           // O(1)       O(n)        Remove element
us.find(x);                            // O(1)       O(n)        Search element
us.count(x);                           // O(1)       O(n)        Check existence

// COMMON USE CASES
// 1. Fast lookup/deduplication
unordered_set<int> seen;
for (int num : nums) {
    if (seen.count(num)) continue;
    seen.insert(num);
}

// 2. Two sum problem
unordered_set<int> seen;
for (int num : nums) {
    if (seen.count(target - num)) return true;
    seen.insert(num);
}

// 3. Finding intersection
unordered_set<int> s1(nums1.begin(), nums1.end());
for (int num : nums2) {
    if (s1.count(num)) result.push_back(num);
}

// 4. Cycle detection
unordered_set<Node*> visited;

// 5. Word break problem
unordered_set<string> wordDict(words.begin(), words.end());
```

### **unordered_map** - Hash Map
```cpp
#include <unordered_map>

// OPERATIONS                           Time Avg    Time Worst  Use Case
um[key] = val;                         // O(1)       O(n)        Insert/Update
um.insert({key, val});                 // O(1)       O(n)        Insert
um.erase(key);                         // O(1)       O(n)        Remove
um.find(key);                          // O(1)       O(n)        Search
um.count(key);                         // O(1)       O(n)        Check existence

// COMMON USE CASES
// 1. Frequency counter (fast)
unordered_map<char, int> freq;
for (char c : str) freq[c]++;

// 2. Caching/Memoization
unordered_map<int, int> memo;
if (memo.count(n)) return memo[n];
memo[n] = fibonacci(n-1) + fibonacci(n-2);

// 3. Graph representation
unordered_map<int, vector<int>> graph;

// 4. Two sum with indices
unordered_map<int, int> map;  // value -> index
for (int i = 0; i < nums.size(); i++) {
    if (map.count(target - nums[i]))
        return {map[target - nums[i]], i};
    map[nums[i]] = i;
}

// 5. Anagram grouping
unordered_map<string, vector<string>> groups;
for (string& str : strs) {
    string key = str;
    sort(key.begin(), key.end());
    groups[key].push_back(str);
}

// 6. LRU Cache
unordered_map<int, list<pair<int,int>>::iterator> cache;
```

## 5. ALGORITHMS

### **Sorting Algorithms**
```cpp
#include <algorithm>

// SORTING OPERATIONS                   Time        Space       Use Case
sort(begin, end);                      // O(nlogn)   O(logn)     General sorting
sort(begin, end, greater<>());         // O(nlogn)   O(logn)     Descending order
stable_sort(begin, end);               // O(nlogn)   O(n)        Preserve relative order
partial_sort(begin, mid, end);         // O(nlogk)   O(1)        Sort first k elements
nth_element(begin, nth, end);          // O(n)       O(1)        Find nth element
is_sorted(begin, end);                 // O(n)       O(1)        Check if sorted

// USE CASES
// 1. Custom sorting
sort(people.begin(), people.end(), [](Person& a, Person& b) {
    return a.age < b.age;  // Sort by age
});

// 2. Find k-th smallest element
nth_element(v.begin(), v.begin() + k - 1, v.end());
int kth_smallest = v[k-1];

// 3. Top K elements
partial_sort(v.begin(), v.begin() + k, v.end(), greater<>());

// 4. Sort with indices
vector<int> idx(n);
iota(idx.begin(), idx.end(), 0);
sort(idx.begin(), idx.end(), [&](int i, int j) {
    return arr[i] < arr[j];
});
```

### **Binary Search Algorithms**
```cpp
// BINARY SEARCH                        Time        Space       Use Case
binary_search(begin, end, val);        // O(logn)    O(1)        Check existence
lower_bound(begin, end, val);          // O(logn)    O(1)        First >= val
upper_bound(begin, end, val);          // O(logn)    O(1)        First > val
equal_range(begin, end, val);          // O(logn)    O(1)        Range of val

// USE CASES
// 1. Find insertion position
auto pos = lower_bound(v.begin(), v.end(), x);
v.insert(pos, x);

// 2. Count occurrences in sorted array
int count = upper_bound(v.begin(), v.end(), x) - 
            lower_bound(v.begin(), v.end(), x);

// 3. Find closest element
auto it = lower_bound(v.begin(), v.end(), target);
if (it != v.begin()) {
    auto prev = it - 1;
    if (abs(*prev - target) < abs(*it - target))
        it = prev;
}

// 4. Binary search on answer
int left = 0, right = 1e9;
while (left < right) {
    int mid = left + (right - left) / 2;
    if (isPossible(mid)) right = mid;
    else left = mid + 1;
}
```

### **Permutation Algorithms**
```cpp
// PERMUTATION                          Time        Space       Use Case
next_permutation(begin, end);          // O(n)       O(1)        Next lexicographic
prev_permutation(begin, end);          // O(n)       O(1)        Previous lexicographic
reverse(begin, end);                   // O(n)       O(1)        Reverse container
rotate(begin, mid, end);               // O(n)       O(1)        Rotate elements
shuffle(begin, end, rng);              // O(n)       O(1)        Random shuffle

// USE CASES
// 1. Generate all permutations
vector<int> v = {1, 2, 3};
do {
    // Process permutation
} while (next_permutation(v.begin(), v.end()));

// 2. Next greater number with same digits
next_permutation(digits.begin(), digits.end());

// 3. Rotate array by k positions
rotate(v.begin(), v.begin() + k, v.end());

// 4. Random shuffle for sampling
random_device rd;
mt19937 g(rd());
shuffle(v.begin(), v.end(), g);
```

### **Numeric Algorithms**
```cpp
#include <numeric>

// NUMERIC OPERATIONS                   Time        Space       Use Case
accumulate(begin, end, init);          // O(n)       O(1)        Sum/Product
inner_product(begin1, end1, begin2, 0);// O(n)       O(1)        Dot product
iota(begin, end, start);               // O(n)       O(1)        Fill with sequence
partial_sum(begin, end, dest);         // O(n)       O(1)        Prefix sum
adjacent_difference(begin, end, dest); // O(n)       O(1)        Differences

// USE CASES
// 1. Calculate sum
int sum = accumulate(v.begin(), v.end(), 0);

// 2. Calculate product
int product = accumulate(v.begin(), v.end(), 1, multiplies<>());

// 3. Dot product of vectors
int dot = inner_product(v1.begin(), v1.end(), v2.begin(), 0);

// 4. Generate indices
vector<int> indices(n);
iota(indices.begin(), indices.end(), 0);  // [0, 1, 2, ..., n-1]

// 5. Prefix sum array
vector<int> prefix(n);
partial_sum(arr.begin(), arr.end(), prefix.begin());

// 6. Calculate differences
vector<int> diff(n);
adjacent_difference(arr.begin(), arr.end(), diff.begin());
```

### **Set Operations** (on sorted ranges)
```cpp
// SET OPERATIONS                       Time        Space       Use Case
set_union(b1, e1, b2, e2, dest);      // O(n+m)     O(1)        Union of sets
set_intersection(b1, e1, b2, e2, dest);// O(n+m)    O(1)        Intersection
set_difference(b1, e1, b2, e2, dest);  // O(n+m)    O(1)        A - B
includes(b1, e1, b2, e2);              // O(n+m)     O(1)        Check subset

// USE CASES
// 1. Find common elements
vector<int> common;
set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(),
                 back_inserter(common));

// 2. Find unique elements in A
vector<int> unique_a;
set_difference(a.begin(), a.end(), b.begin(), b.end(),
               back_inserter(unique_a));

// 3. Merge two sorted arrays
vector<int> merged;
set_union(v1.begin(), v1.end(), v2.begin(), v2.end(),
          back_inserter(merged));

// 4. Check if one set contains another
bool is_subset = includes(v1.begin(), v1.end(), v2.begin(), v2.end());
```

### **Heap Operations**
```cpp
// HEAP OPERATIONS                      Time        Space       Use Case
make_heap(begin, end);                 // O(n)       O(1)        Convert to heap
push_heap(begin, end);                 // O(logn)    O(1)        Add to heap
pop_heap(begin, end);                  // O(logn)    O(1)        Remove from heap
sort_heap(begin, end);                 // O(nlogn)   O(1)        Heap sort

// USE CASES
// 1. Custom priority queue
vector<int> heap = {3, 1, 4, 1, 5};
make_heap(heap.begin(), heap.end());

// Add element
heap.push_back(2);
push_heap(heap.begin(), heap.end());

// Remove max
pop_heap(heap.begin(), heap.end());
int max_val = heap.back();
heap.pop_back();

// 2. K largest elements
make_heap(v.begin(), v.end());
for (int i = 0; i < k; i++) {
    pop_heap(v.begin(), v.end() - i);
}
// Last k elements are k largest

// 3. Running median
vector<int> maxHeap, minHeap;
make_heap(maxHeap.begin(), maxHeap.end());
make_heap(minHeap.begin(), minHeap.end(), greater<>());
```

### **Search Algorithms**
```cpp
// SEARCH OPERATIONS                    Time        Space       Use Case
find(begin, end, val);                 // O(n)       O(1)        Find first occurrence
find_if(begin, end, pred);             // O(n)       O(1)        Find by condition
count(begin, end, val);                // O(n)       O(1)        Count occurrences
count_if(begin, end, pred);            // O(n)       O(1)        Count by condition
search(b1, e1, b2, e2);               // O(n*m)     O(1)        Find subsequence

// USE CASES
// 1. Find element
auto it = find(v.begin(), v.end(), target);
if (it != v.end()) {
    int index = it - v.begin();
}

// 2. Find first even number
auto it = find_if(v.begin(), v.end(), [](int x) { 
    return x % 2 == 0; 
});

// 3. Count specific value
int cnt = count(v.begin(), v.end(), target);

// 4. Count elements satisfying condition
int cnt = count_if(v.begin(), v.end(), [](int x) { 
    return x > 10; 
});

// 5. Find pattern in sequence
vector<int> pattern = {1, 2, 3};
auto it = search(v.begin(), v.end(), pattern.begin(), pattern.end());
```

### **Min/Max Operations**
```cpp
// MIN/MAX OPERATIONS                   Time        Space       Use Case
min_element(begin, end);               // O(n)       O(1)        Find minimum
max_element(begin, end);               // O(n)       O(1)        Find maximum
minmax_element(begin, end);            // O(n)       O(1)        Find both

// USE CASES
// 1. Find min/max in range
auto min_it = min_element(v.begin(), v.end());
auto max_it = max_element(v.begin(), v.end());

// 2. Find range of values
auto [min_it, max_it] = minmax_element(v.begin(), v.end());
int range = *max_it - *min_it;

// 3. Custom comparison
auto it = max_element(people.begin(), people.end(),
    [](const Person& a, const Person& b) {
        return a.salary < b.salary;
    });
```

### **Modifying Operations**
```cpp
// MODIFY OPERATIONS                    Time        Space       Use Case
copy(begin, end, dest);                // O(n)       O(1)        Copy elements
move(begin, end, dest);                // O(n)       O(1)        Move elements
transform(begin, end, dest, op);       // O(n)       O(1)        Apply function
replace(begin, end, old, new);         // O(n)       O(1)        Replace values
remove(begin, end, val);               // O(n)       O(1)        Remove values
unique(begin, end);                    // O(n)       O(1)        Remove duplicates

// USE CASES
// 1. Copy to another container
vector<int> dest(v.size());
copy(v.begin(), v.end(), dest.begin());

// 2. Transform elements
transform(v.begin(), v.end(), v.begin(), 
    [](int x) { return x * 2; });  // Double all values

// 3. Replace all occurrences
replace(v.begin(), v.end(), old_val, new_val);

// 4. Remove element (erase-remove idiom)
v.erase(remove(v.begin(), v.end(), val), v.end());

// 5. Remove consecutive duplicates
v.erase(unique(v.begin(), v.end()), v.end());

// 6. Apply operation on two vectors
transform(v1.begin(), v1.end(), v2.begin(), result.begin(), plus<>());
```

## 6. STRING OPERATIONS

```cpp
#include <string>

// STRING OPERATIONS                    Time        Space       Use Case
s.append(str);                         // O(n)       O(1)*       Concatenate
s.insert(pos, str);                    // O(n)       O(1)        Insert at position
s.erase(pos, len);                     // O(n)       O(1)        Remove substring
s.replace(pos, len, str);              // O(n)       O(1)        Replace part
s.substr(pos, len);                    // O(n)       O(n)        Extract substring
s.find(str);                           // O(n*m)     O(1)        Find substring
s.rfind(str);                          // O(n*m)     O(1)        Find from end
s.find_first_of(chars);                // O(n*m)     O(1)        Find any of chars
s.find_last_of(chars);                 // O(n*m)     O(1)        Find any from end

// USE CASES
// 1. String concatenation
string result = s1 + s2;
s1.append(s2);  // More efficient

// 2. Find all occurrences
size_t pos = s.find(pattern);
while (pos != string::npos) {
    positions.push_back(pos);
    pos = s.find(pattern, pos + 1);
}

// 3. Replace all occurrences
size_t pos = 0;
while ((pos = s.find(old_str, pos)) != string::npos) {
    s.replace(pos, old_str.length(), new_str);
    pos += new_str.length();
}

// 4. Split string
stringstream ss(s);
string token;
while (getline(ss, token, delimiter)) {
    tokens.push_back(token);
}

// 5. Check prefix/suffix
bool hasPrefix = s.find(prefix) == 0;
bool hasSuffix = s.rfind(suffix) == s.length() - suffix.length();

// 6. Case conversion
transform(s.begin(), s.end(), s.begin(), ::tolower);
transform(s.begin(), s.end(), s.begin(), ::toupper);
```

## 7. SPECIAL CONTAINERS

### **bitset** - Fixed-Size Bit Array
```cpp
#include <bitset>

// BITSET OPERATIONS                    Time        Space       Use Case
bs.set();                              // O(n)       O(1)        Set all bits to 1
bs.set(pos);                           // O(1)       O(1)        Set specific bit
bs.reset();                            // O(n)       O(1)        Set all bits to 0
bs.reset(pos);                         // O(1)       O(1)        Clear specific bit
bs.flip();                             // O(n)       O(1)        Flip all bits
bs.flip(pos);                          // O(1)       O(1)        Flip specific bit
bs.test(pos);                          // O(1)       O(1)        Check bit value
bs.count();                            // O(n)       O(1)        Count set bits

// USE CASES
// 1. Sieve of Eratosthenes
bitset<1000001> isPrime;
isPrime.set();  // All true initially
isPrime[0] = isPrime[1] = 0;
for (int i = 2; i * i <= n; i++) {
    if (isPrime[i]) {
        for (int j = i * i; j <= n; j += i)
            isPrime[j] = 0;
    }
}

// 2. Subset generation
bitset<20> mask;
for (int i = 0; i < (1 << n); i++) {
    mask = i;
    // Process subset represented by mask
}

// 3. Binary operations
bitset<8> a("10101010");
bitset<8> b("11001100");
auto c = a & b;  // AND
auto d = a | b;  // OR
auto e = a ^ b;  // XOR

// 4. Memory-efficient boolean array
bitset<1000000> visited;  // Only 125KB vs 1MB for bool array

// 5. Fast bit manipulation
bitset<32> num(42);
int leading_zeros = num._Find_first();
```

### **pair** - Two Elements
```cpp
#include <utility>

// PAIR OPERATIONS                      Time        Space       Use Case
pair<int, string> p = {1, "hello"};
p.first;                               // O(1)       O(1)        Access first
p.second;                              // O(1)       O(1)        Access second

// USE CASES
// 1. Return multiple values
pair<int, int> findMinMax(vector<int>& v) {
    return {*min_element(v.begin(), v.end()),
            *max_element(v.begin(), v.end())};
}

// 2. Store coordinates
vector<pair<int, int>> points = {{0,0}, {1,1}, {2,2}};

// 3. Map with pair as key
map<pair<int, int>, string> grid;
grid[{x, y}] = "occupied";

// 4. Priority queue with priority
priority_queue<pair<int, int>> pq;  // {priority, value}
pq.push({5, 100});

// 5. Graph edges with weights
vector<pair<int, int>> adj[n];  // {neighbor, weight}
```

### **tuple** - Multiple Elements
```cpp
#include <tuple>

// TUPLE OPERATIONS                     Time        Space       Use Case
tuple<int, string, double> t = {1, "hello", 3.14};
get<0>(t);                             // O(1)       O(1)        Access by index
tie(a, b, c) = t;                      // O(1)       O(1)        Unpack

// USE CASES
// 1. Return multiple values
tuple<int, int, int> getDate() {
    return {day, month, year};
}
auto [d, m, y] = getDate();  // C++17

// 2. Store records
vector<tuple<string, int, double>> students;
// {name, id, gpa}

// 3. Priority queue with multiple criteria
priority_queue<tuple<int, int, string>> pq;
// {priority1, priority2, data}

// 4. 3D coordinates
using Point3D = tuple<int, int, int>;
Point3D p = {x, y, z};
```

## Common Problem Patterns

### Pattern 1: **Two Pointers + Container**
```cpp
// Sliding window with deque
deque<int> window;
for (int i = 0; i < n; i++) {
    while (!window.empty() && window.front() < i - k)
        window.pop_front();
    while (!window.empty() && arr[window.back()] < arr[i])
        window.pop_back();
    window.push_back(i);
}

// Two sum with hash map
unordered_map<int, int> seen;
for (int i = 0; i < n; i++) {
    if (seen.count(target - nums[i]))
        return {seen[target - nums[i]], i};
    seen[nums[i]] = i;
}
```

### Pattern 2: **Graph Algorithms**
```cpp
// BFS with queue
queue<int> q;
vector<bool> visited(n);
q.push(start);
visited[start] = true;
while (!q.empty()) {
    int node = q.front(); q.pop();
    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            visited[neighbor] = true;
            q.push(neighbor);
        }
    }
}

// Dijkstra with priority queue
priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
vector<int> dist(n, INT_MAX);
pq.push({0, start});
dist[start] = 0;
while (!pq.empty()) {
    auto [d, u] = pq.top(); pq.pop();
    if (d > dist[u]) continue;
    for (auto [v, w] : adj[u]) {
        if (dist[u] + w < dist[v]) {
            dist[v] = dist[u] + w;
            pq.push({dist[v], v});
        }
    }
}
```

### Pattern 3: **Dynamic Programming**
```cpp
// Memoization with map
unordered_map<string, int> memo;
function<int(int, int)> dp = [&](int i, int j) {
    string key = to_string(i) + "," + to_string(j);
    if (memo.count(key)) return memo[key];
    // Compute result
    return memo[key] = result;
};

// Tabulation with vector
vector<vector<int>> dp(n, vector<int>(m, 0));
for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
        // Fill dp table
    }
}
```

## Performance Guidelines

### Container Selection:
- **Random access needed**: `vector`, `deque`, `array`
- **Frequent insertion/deletion**: `list`, `unordered_map`
- **Sorted data**: `set`, `map`
- **Fast average lookup**: `unordered_set`, `unordered_map`
- **Stack/Queue operations**: `stack`, `queue`, `deque`
- **Priority ordering**: `priority_queue`, `multiset`

### Optimization Tips:
1. **Reserve capacity**: `v.reserve(n)` for vectors when size is known
2. **Use emplace**: `v.emplace_back()` instead of `push_back()` for objects
3. **Choose right container**: Hash tables for average O(1), trees for worst-case O(log n)
4. **Avoid unnecessary copies**: Use references and move semantics
5. **Batch operations**: Sort once instead of maintaining sorted order
6. **Cache-friendly**: Prefer `vector` over `list` for better cache locality

This comprehensive guide covers all major STL components with their practical applications. Each operation includes its use case to help you choose the right tool for your specific problem.##