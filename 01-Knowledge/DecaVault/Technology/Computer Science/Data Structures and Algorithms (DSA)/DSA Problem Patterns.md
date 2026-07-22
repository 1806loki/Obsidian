| Pattern                     | When It Appears      | Keywords / Clues                                     |
| --------------------------- | -------------------- | ---------------------------------------------------- |
| **Sliding Window**          | Subarray / substring | “longest”, “shortest”, “contiguous”, “at most K”     |
| **Two Pointers**            | Arrays / strings     | “sorted”, “pair”, “remove duplicates”, “in-place”    |
| **Hashing / Frequency Map** | Counting / lookup    | “frequency”, “anagram”, “unique”, “first occurrence” |
| **Prefix Sum**              | Range queries        | “subarray sum equals K”, “range sum”                 |
| **Binary Search**           | Sorted / monotonic   | “minimum X”, “maximum X”, “can we…?”                 |
| **DFS / BFS**               | Trees / graphs       | “connected”, “path”, “distance”, “levels”            |
| **Backtracking**            | Combinations         | “all possible”, “generate”, “permutations”           |
| **Dynamic Programming**     | Optimization         | “maximum”, “minimum”, “number of ways”               |
| **Greedy**                  | Optimal choice       | “locally optimal”, “earliest”, “minimum steps”       |
| **Heap / Priority Queue**   | Top K                | “K largest”, “K smallest”, “stream”                  |
| **Monotonic Stack**         | Next greater         | “next greater”, “previous smaller”                   |
| **Union Find**              | Connectivity         | “number of components”, “cycles”                     |


### Step 1: Scan the Constraints (Most Important)

|Constraint|Meaning|
|---|---|
|`n ≤ 10^5`|O(n) or O(n log n) only|
|`n ≤ 10^3`|DP or O(n²) possible|
|“Streaming input”|Heap / sliding window|
|“Sorted array”|Binary search / two pointers|
