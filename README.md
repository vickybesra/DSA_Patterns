# Data Structures and Algorithms (DSA) Repository

A comprehensive collection of C++ templates, patterns, and implementations for competitive programming and technical interviews. This repository contains ready-to-use code templates covering major DSA patterns commonly found in LeetCode and coding interviews.

## 📁 Repository Structure

```
dsa/
├── README.md                           # This file
├── pattern2.cpp                        # Complete pattern templates (comprehensive)
├── patterns.cpp                        # Compact 32 core patterns template pack
├── stl_syntax.cpp                      # STL (Standard Template Library) syntax reference
├── Leetcode Pattern Recognition Guide.pdf  # Pattern recognition guide
│
├── graph/
│   ├── bfs.cpp                        # Breadth-First Search implementation
│   └── implementation.cpp             # Graph implementation examples
│
├── string/
│   └── basics_implementation.cpp      # String manipulation basics
│
└── two_pointers/
    ├── remove_duplicates_from_shorted_arrays.cpp  # Remove duplicates pattern
    └── two_sum.cpp                    # Two Sum problem implementation
```

## 📚 File Descriptions

### Core Pattern Files

#### `pattern2.cpp`
**Complete Pattern Templates for LeetCode Problems (C++)**

A comprehensive collection of DSA patterns covering:
- **Data Structure Patterns**: Arrays, Linked Lists, Stacks, Queues, Hash Tables, Trees, Graphs, Tries, Union-Find
- **Algorithm Patterns**: Binary Search, Dynamic Programming, Backtracking, Greedy, Bit Manipulation, Graph Algorithms

**Features:**
- Uses modern C++ features (C++11/14/17)
- Includes necessary headers
- Follows C++ best practices
- Optimized for performance
- Handles edge cases properly

**Patterns Included:**
1. Array Patterns (Two Pointers, Sliding Window, Prefix Sum, Cyclic Sort, Matrix)
2. Linked List Patterns (Fast & Slow Pointers, Reversal, Merging)
3. Stack Patterns (Monotonic Stack, Expression Evaluation)
4. Queue/Heap Patterns (Two Heaps, Top K, K-way Merge)
5. Hash Table Patterns (Frequency Counting, Two Sum, LRU Cache)
6. Tree Patterns (BFS, DFS, Path Sum, BST Validation)
7. Graph Patterns (DFS, BFS, Shortest Path, Cycle Detection, Topological Sort)
8. Trie Pattern
9. Union-Find Pattern
10. Binary Search Patterns
11. Dynamic Programming Patterns
12. Backtracking Patterns
13. Greedy Patterns
14. Bit Manipulation Patterns
15. Advanced Graph Algorithms (Dijkstra, Bellman-Ford, Kruskal, Floyd-Warshall)

#### `patterns.cpp`
**Compact 32 Core Patterns Template Pack**

A condensed version covering ~80-90% of interview DSA questions:
- Uses `<bits/stdc++.h>` for brevity
- Self-contained functions
- Ready-to-use templates
- Covers essential patterns for quick reference

#### `stl_syntax.cpp`
**STL Syntax Reference**

Quick reference guide for Standard Template Library:
- Container operations
- Algorithm functions
- Iterator usage
- Common STL patterns and idioms

### Topic-Specific Implementations

#### `graph/`
- **`bfs.cpp`**: Breadth-First Search implementation with examples
- **`implementation.cpp`**: Graph representation and basic operations

#### `string/`
- **`basics_implementation.cpp`**: String manipulation fundamentals and common operations

#### `two_pointers/`
- **`remove_duplicates_from_shorted_arrays.cpp`**: Remove duplicates from sorted arrays using two pointers
- **`two_sum.cpp`**: Two Sum problem implementation

## 🚀 How to Use

### For Competitive Programming
1. Copy the relevant pattern template from `pattern2.cpp` or `patterns.cpp`
2. Adapt the template to your specific problem
3. Test with sample inputs

### For Interview Preparation
1. Study the patterns in `pattern2.cpp` for comprehensive understanding
2. Use `patterns.cpp` for quick reference during practice
3. Refer to topic-specific files for detailed implementations

### Compilation
```bash
# Compile any C++ file
g++ -std=c++17 filename.cpp -o output

# Example
g++ -std=c++17 pattern2.cpp -o pattern2
./pattern2
```

## 📖 Pattern Categories

### Data Structure Patterns
- ✅ Arrays & Matrices
- ✅ Linked Lists
- ✅ Stacks & Queues
- ✅ Hash Tables
- ✅ Trees (Binary Trees, BST)
- ✅ Graphs
- ✅ Tries
- ✅ Union-Find

### Algorithm Patterns
- ✅ Two Pointers
- ✅ Sliding Window
- ✅ Binary Search
- ✅ Dynamic Programming
- ✅ Backtracking
- ✅ Greedy Algorithms
- ✅ Bit Manipulation
- ✅ Graph Algorithms (DFS, BFS, Shortest Path, MST)

## 💡 Key Features

- **Modern C++**: Uses C++11/14/17 features
- **Well-Documented**: Inline comments explain logic
- **Production-Ready**: Handles edge cases
- **Optimized**: Performance-focused implementations
- **Comprehensive**: Covers 90%+ of interview patterns

## 📝 Notes

- All code uses `using namespace std;` for brevity (remove if needed for production)
- Some files use `<bits/stdc++.h>` (GCC-specific, replace with specific headers if needed)
- Templates are designed to be easily adaptable to specific problems
- Memory management is handled appropriately (RAII principles where applicable)

## 🔗 Additional Resources

- **Leetcode Pattern Recognition Guide.pdf**: Comprehensive guide for pattern recognition
- Each pattern includes multiple variations and examples
- Code follows competitive programming best practices

## 📄 License

This repository is for educational and interview preparation purposes.

## 🤝 Contributing

Feel free to add more patterns, optimizations, or improvements to existing code.

---

**Happy Coding! 🎯**

