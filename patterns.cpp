// Here’s a compact C++ template pack for the 32 core patterns that cover ~80–90% of interview DSA questions. You can copy/paste functions as needed.

// Note:
// - Uses <bits/stdc++.h> for brevity; replace with specific headers if required.
// - Function names are self-contained; no main() needed.

// ```cpp
#include <bits/stdc++.h>
using namespace std;

/********** 1) TWO POINTERS (incl. fast–slow) **********/
vector<int> twoSumTwoPointersSorted(const vector<int>& a, int target) {
    int i = 0, j = (int)a.size() - 1;
    while (i < j) {
        long long s = (long long)a[i] + a[j];
        if (s == target) return {i, j};
        (s < target) ? ++i : --j;
    }
    return {-1, -1};
}
int removeDuplicatesSorted(vector<int>& a) { // keeps 1 copy
    int k = 0;
    for (int x : a) if (k == 0 || a[k-1] != x) a[k++] = x;
    return k;
}

/********** 2) SLIDING WINDOW (fixed/variable) **********/
int maxSumFixedWindow(const vector<int>& a, int k) {
    long long cur = 0, best = LLONG_MIN;
    for (int i = 0; i < a.size(); ++i) {
        cur += a[i];
        if (i >= k) cur -= a[i-k];
        if (i >= k-1) best = max(best, cur);
    }
    return (int)best;
}
int lengthOfLongestSubstringNoRepeat(const string& s) {
    vector<int> last(256, -1);
    int start = 0, best = 0;
    for (int i = 0; i < s.size(); ++i) {
        if (last[s[i]] >= start) start = last[s[i]] + 1;
        last[s[i]] = i;
        best = max(best, i - start + 1);
    }
    return best;
}

/********** 3) PREFIX SUMS / DIFFERENCE ARRAY **********/
vector<long long> buildPrefix(const vector<int>& a) {
    vector<long long> p(a.size()+1, 0);
    for (int i = 0; i < a.size(); ++i) p[i+1] = p[i] + a[i];
    return p;
}
long long rangeSum(const vector<long long>& p, int l, int r) { // [l,r]
    return p[r+1] - p[l];
}
vector<int> applyRangeIncrements(int n, const vector<array<int,3>>& updates) {
    vector<int> diff(n+1, 0);
    for (auto [l,r,delta] : updates) diff[l] += delta, diff[r+1] -= delta;
    vector<int> a(n, 0);
    int cur = 0; for (int i = 0; i < n; ++i) a[i] = (cur += diff[i]);
    return a;
}

/********** 4) HASHING / FREQ MAPS (2-sum, anagram) **********/
pair<int,int> twoSum(const vector<int>& a, int target) {
    unordered_map<int,int> pos;
    for (int i = 0; i < a.size(); ++i) {
        int need = target - a[i];
        if (pos.count(need)) return {pos[need], i};
        pos[a[i]] = i;
    }
    return {-1,-1};
}
bool isAnagram(const string& s, const string& t) {
    if (s.size() != t.size()) return false;
    array<int,26> cnt{}; 
    for (char c: s) cnt[c-'a']++;
    for (char c: t) if (--cnt[c-'a'] < 0) return false;
    return true;
}

/********** 5) KADANE (max subarray, circular) **********/
int maxSubArrayKadane(const vector<int>& a) {
    long long best = LLONG_MIN, cur = 0;
    for (int x : a) { cur = max<long long>(x, cur + x); best = max(best, cur); }
    return (int)best;
}
int maxSubarrayCircular(const vector<int>& a) {
    int total = accumulate(a.begin(), a.end(), 0);
    // max subarray
    int maxEnd = 0, maxSoFar = INT_MIN;
    for (int x: a) { maxEnd = max(x, maxEnd + x); maxSoFar = max(maxSoFar, maxEnd); }
    // min subarray
    int minEnd = 0, minSoFar = INT_MAX;
    for (int x: a) { minEnd = min(x, minEnd + x); minSoFar = min(minSoFar, minEnd); }
    if (minSoFar == total) return maxSoFar; // all negative
    return max(maxSoFar, total - minSoFar);
}

/********** 6) MONOTONIC STACK/DEQUE **********/
vector<int> nextGreaterElements(const vector<int>& a) {
    vector<int> res(a.size(), -1); stack<int> st;
    for (int i = 0; i < a.size(); ++i) {
        while (!st.empty() && a[st.top()] < a[i]) { res[st.top()] = i; st.pop(); }
        st.push(i);
    }
    return res; // indices of next greater; -1 if none
}
vector<int> slidingWindowMax(const vector<int>& a, int k) {
    deque<int> dq; vector<int> out;
    for (int i = 0; i < a.size(); ++i) {
        if (!dq.empty() && dq.front() <= i-k) dq.pop_front();
        while (!dq.empty() && a[dq.back()] <= a[i]) dq.pop_back();
        dq.push_back(i);
        if (i >= k-1) out.push_back(a[dq.front()]);
    }
    return out;
}

/********** 7) BINARY SEARCH VARIANTS **********/
int lowerBoundIdx(const vector<int>& a, int target) {
    int l = 0, r = (int)a.size(); // [l,r)
    while (l < r) {
        int m = l + (r-l)/2;
        if (a[m] < target) l = m+1; else r = m;
    }
    return l;
}
int upperBoundIdx(const vector<int>& a, int target) { // first > target
    int l = 0, r = (int)a.size();
    while (l < r) {
        int m = l + (r-l)/2;
        if (a[m] <= target) l = m+1; else r = m;
    }
    return l;
}
int searchRotated(const vector<int>& a, int target) {
    int l = 0, r = (int)a.size()-1;
    while (l <= r) {
        int m = l + (r-l)/2;
        if (a[m] == target) return m;
        if (a[l] <= a[m]) { // left sorted
            if (a[l] <= target && target < a[m]) r = m-1; else l = m+1;
        } else { // right sorted
            if (a[m] < target && target <= a[r]) l = m+1; else r = m-1;
        }
    }
    return -1;
}

/********** 8) BINARY SEARCH ON ANSWER (monotonic predicate) **********/
template <class F>
long long firstTrue(long long lo, long long hi, F pred) { // [lo,hi]
    while (lo < hi) {
        long long mid = lo + (hi - lo)/2;
        if (pred(mid)) hi = mid; else lo = mid + 1;
    }
    return lo;
}
// Example: minimize capacity to ship within D days
bool canShip(const vector<int>& w, int days, int cap) {
    int d = 1, cur = 0;
    for (int x : w) {
        if (x > cap) return false;
        if (cur + x > cap) { d++; cur = 0; }
        cur += x;
    }
    return d <= days;
}
int minShipCapacity(const vector<int>& w, int days) {
    int lo = *max_element(w.begin(), w.end());
    int hi = accumulate(w.begin(), w.end(), 0);
    return (int)firstTrue(lo, hi, [&](int m){ return canShip(w, days, m); });
}

/********** 9) QUICKSELECT (kth smallest / largest) **********/
int partitionIdx(vector<int>& a, int l, int r) {
    int pivot = a[r], i = l;
    for (int j = l; j < r; ++j) if (a[j] <= pivot) swap(a[i++], a[j]);
    swap(a[i], a[r]); return i;
}
int quickselect(vector<int>& a, int k) { // kth smallest, k in [0,n-1]
    int l = 0, r = (int)a.size()-1;
    while (l <= r) {
        int p = partitionIdx(a, l, r);
        if (p == k) return a[p];
        (p < k) ? l = p+1 : r = p-1;
    }
    return -1;
}

/********** 10) INTERVALS: MERGE / INSERT **********/
vector<vector<int>> mergeIntervals(vector<vector<int>> iv) {
    if (iv.empty()) return {};
    sort(iv.begin(), iv.end());
    vector<vector<int>> res; res.push_back(iv[0]);
    for (int i = 1; i < iv.size(); ++i) {
        if (iv[i][0] <= res.back()[1]) res.back()[1] = max(res.back()[1], iv[i][1]);
        else res.push_back(iv[i]);
    }
    return res;
}
vector<vector<int>> insertInterval(vector<vector<int>> iv, vector<int> x) {
    iv.push_back(x); return mergeIntervals(iv);
}

/********** 11) SWEEP-LINE: MIN ROOMS / OVERLAPS **********/
int minMeetingRooms(vector<vector<int>> iv) {
    vector<pair<int,int>> ev;
    for (auto& v : iv) ev.push_back({v[0], +1}), ev.push_back({v[1], -1});
    sort(ev.begin(), ev.end());
    int cur = 0, best = 0;
    for (auto& [t, d] : ev) { cur += d; best = max(best, cur); }
    return best;
}

/********** 12) LINKED LIST REVERSAL (full/sublist/k-group) **********/
struct ListNode { int val; ListNode* next; ListNode(int x): val(x), next(nullptr) {} };
ListNode* reverseList(ListNode* head) {
    ListNode *prev = nullptr, *cur = head;
    while (cur) { auto nxt = cur->next; cur->next = prev; prev = cur; cur = nxt; }
    return prev;
}
ListNode* reverseBetween(ListNode* head, int m, int n) {
    if (!head || m == n) return head;
    ListNode dummy(0); dummy.next = head; ListNode* pre = &dummy;
    for (int i = 1; i < m; ++i) pre = pre->next;
    ListNode* cur = pre->next;
    for (int i = 0; i < n-m; ++i) {
        ListNode* move = cur->next; cur->next = move->next;
        move->next = pre->next; pre->next = move;
    }
    return dummy.next;
}
ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode* cur = head; int cnt = 0;
    while (cur && cnt < k) { cur = cur->next; cnt++; }
    if (cnt < k) return head;
    cur = head; ListNode *prev = nullptr, *nxt = nullptr;
    for (int i = 0; i < k; ++i) { nxt = cur->next; cur->next = prev; prev = cur; cur = nxt; }
    head->next = reverseKGroup(cur, k);
    return prev;
}

/********** 13) FAST–SLOW: CYCLE / MIDDLE **********/
bool hasCycle(ListNode* head) {
    ListNode *s = head, *f = head;
    while (f && f->next) { s = s->next; f = f->next->next; if (s == f) return true; }
    return false;
}
ListNode* detectCycleStart(ListNode* head) {
    ListNode *s = head, *f = head;
    while (f && f->next) { s = s->next; f = f->next->next; if (s == f) break; }
    if (!f || !f->next) return nullptr;
    s = head; while (s != f) s = s->next, f = f->next;
    return s;
}
ListNode* middleNode(ListNode* head) {
    ListNode *s = head, *f = head;
    while (f && f->next) s = s->next, f = f->next->next;
    return s;
}

/********** 14) TREE TRAVERSALS (DFS/BFS) **********/
struct TreeNode { int val; TreeNode* left; TreeNode* right; TreeNode(int v):val(v),left(nullptr),right(nullptr){} };
void inorder(TreeNode* root, vector<int>& out) {
    if (!root) return; inorder(root->left, out); out.push_back(root->val); inorder(root->right, out);
}
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> res; if (!root) return res;
    queue<TreeNode*> q; q.push(root);
    while (!q.empty()) {
        int sz = q.size(); res.push_back({});
        while (sz--) {
            auto* n = q.front(); q.pop(); res.back().push_back(n->val);
            if (n->left) q.push(n->left); if (n->right) q.push(n->right);
        }
    }
    return res;
}

/********** 15) DEPTH / DIAMETER / PATH SUM **********/
int maxDepth(TreeNode* root) { return root ? 1 + max(maxDepth(root->left), maxDepth(root->right)) : 0; }
int diameterOfBinaryTree(TreeNode* root) {
    int best = 0;
    function<int(TreeNode*)> h = [&](TreeNode* n){
        if (!n) return 0;
        int L = h(n->left), R = h(n->right);
        best = max(best, L + R);
        return 1 + max(L, R);
    };
    h(root); return best;
}
// Count paths sum to target (prefix-sum on tree)
int pathSumCount(TreeNode* root, int target) {
    unordered_map<long long,int> cnt; cnt[0] = 1; int ans = 0;
    function<void(TreeNode*, long long)> dfs = [&](TreeNode* n, long long s){
        if (!n) return;
        s += n->val; ans += cnt[s - target];
        cnt[s]++; dfs(n->left, s); dfs(n->right, s); cnt[s]--;
    };
    dfs(root, 0); return ans;
}

/********** 16) LCA (Binary Tree) **********/
TreeNode* LCA(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    TreeNode* L = LCA(root->left, p, q), *R = LCA(root->right, p, q);
    if (L && R) return root; return L ? L : R;
}

/********** 17) BST INVARIANTS (validate / inorder) **********/
bool validateBST(TreeNode* root, long long low = LLONG_MIN, long long high = LLONG_MAX) {
    if (!root) return true;
    if (root->val <= low || root->val >= high) return false;
    return validateBST(root->left, low, root->val) && validateBST(root->right, root->val, high);
}
bool searchBST(TreeNode* root, int target) {
    while (root) {
        if (root->val == target) return true;
        root = target < root->val ? root->left : root->right;
    }
    return false;
}

/********** 18) BFS SHORTEST PATH (graph) + GRID BFS **********/
vector<int> bfsShortestPath(int n, const vector<vector<int>>& adj, int src) {
    vector<int> dist(n, INT_MAX); queue<int> q; dist[src] = 0; q.push(src);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v: adj[u]) if (dist[v] == INT_MAX) dist[v] = dist[u] + 1, q.push(v);
    }
    return dist;
}
int shortestPathInGrid(vector<vector<int>>& g, pair<int,int> s, pair<int,int> t) {
    int n = g.size(), m = g[0].size();
    vector<vector<int>> dist(n, vector<int>(m, -1));
    queue<pair<int,int>> q; q.push(s); dist[s.first][s.second] = 0;
    int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    while (!q.empty()) {
        auto [x,y] = q.front(); q.pop();
        if (make_pair(x,y) == t) return dist[x][y];
        for (auto& d : dirs) {
            int nx = x + d[0], ny = y + d[1];
            if (0<=nx&&nx<n&&0<=ny&&ny<m && g[nx][ny]==1 && dist[nx][ny]==-1) {
                dist[nx][ny] = dist[x][y] + 1; q.push({nx,ny});
            }
        }
    }
    return -1;
}

/********** 19) DFS COMPONENTS + CYCLE DETECTION **********/
int countComponents(int n, const vector<vector<int>>& adj) {
    vector<int> vis(n, 0); int comps = 0;
    function<void(int)> dfs = [&](int u){ vis[u]=1; for (int v: adj[u]) if (!vis[v]) dfs(v); };
    for (int i = 0; i < n; ++i) if (!vis[i]) comps++, dfs(i);
    return comps;
}
// Cycle detection (directed): 0=unvis,1=visiting,2=done
bool hasCycleDirected(int n, const vector<vector<int>>& adj) {
    vector<int> state(n,0);
    function<bool(int)> dfs = [&](int u){
        state[u]=1;
        for (int v: adj[u]) {
            if (state[v]==1) return true;
            if (state[v]==0 && dfs(v)) return true;
        }
        state[u]=2; return false;
    };
    for (int i=0;i<n;++i) if (state[i]==0 && dfs(i)) return true;
    return false;
}
// Cycle detection (undirected)
bool hasCycleUndirected(int n, const vector<vector<int>>& adj) {
    vector<int> vis(n,0);
    function<bool(int,int)> dfs = [&](int u,int p){
        vis[u]=1;
        for (int v: adj[u]) {
            if (v==p) continue;
            if (vis[v]) return true;
            if (dfs(v,u)) return true;
        }
        return false;
    };
    for (int i=0;i<n;++i) if (!vis[i] && dfs(i,-1)) return true;
    return false;
}

/********** 20) TOPOLOGICAL SORT (Kahn + DFS) **********/
vector<int> topoKahn(int n, const vector<vector<int>>& adj) {
    vector<int> indeg(n,0); for (int u=0;u<n;++u) for (int v: adj[u]) indeg[v]++;
    queue<int> q; for (int i=0;i<n;++i) if (!indeg[i]) q.push(i);
    vector<int> order;
    while (!q.empty()) {
        int u=q.front(); q.pop(); order.push_back(u);
        for (int v: adj[u]) if (--indeg[v]==0) q.push(v);
    }
    return order; // if size<n, there is a cycle
}
void topoDFSUtil(int u, const vector<vector<int>>& adj, vector<int>& vis, vector<int>& out) {
    vis[u]=1; for (int v: adj[u]) if (!vis[v]) topoDFSUtil(v,adj,vis,out); out.push_back(u);
}
vector<int> topoDFS(int n, const vector<vector<int>>& adj) {
    vector<int> vis(n,0), out; out.reserve(n);
    for (int i=0;i<n;++i) if (!vis[i]) topoDFSUtil(i,adj,vis,out);
    reverse(out.begin(), out.end()); return out;
}

/********** 21) DIJKSTRA + 0–1 BFS **********/
vector<long long> dijkstra(int n, const vector<vector<pair<int,int>>>& adj, int src) {
    const long long INF = (1LL<<60);
    vector<long long> dist(n, INF);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[src]=0; pq.push({0,src});
    while(!pq.empty()){
        auto [d,u]=pq.top(); pq.pop();
        if (d!=dist[u]) continue;
        for (auto [v,w]: adj[u]) if (dist[v] > d + w) { dist[v] = d + w; pq.push({dist[v], v}); }
    }
    return dist;
}
vector<int> zeroOneBFS(int n, const vector<vector<pair<int,int>>>& adj, int src) {
    deque<int> dq; vector<int> dist(n, INT_MAX);
    dist[src]=0; dq.push_front(src);
    while(!dq.empty()){
        int u=dq.front(); dq.pop_front();
        for (auto [v,w]: adj[u]) {
            int nd = dist[u] + w;
            if (nd < dist[v]) {
                dist[v] = nd;
                (w==0) ? dq.push_front(v) : dq.push_back(v);
            }
        }
    }
    return dist;
}

/********** 22) DISJOINT SET UNION (Union-Find) **********/
struct DSU {
    vector<int> p, r, sz;
    DSU(int n=0){init(n);}
    void init(int n){ p.resize(n); r.assign(n,0); sz.assign(n,1); iota(p.begin(),p.end(),0); }
    int find(int x){ return p[x]==x?x:p[x]=find(p[x]); }
    bool unite(int a,int b){
        a=find(a); b=find(b); if (a==b) return false;
        if (r[a]<r[b]) swap(a,b); p[b]=a; sz[a]+=sz[b]; if (r[a]==r[b]) r[a]++;
        return true;
    }
    int size(int x){ return sz[find(x)]; }
};

/********** 23) BACKTRACKING: SUBSETS / COMBINATIONS **********/
vector<vector<int>> subsets(vector<int> nums) {
    vector<vector<int>> res; vector<int> cur;
    function<void(int)> dfs = [&](int i){
        if (i == nums.size()) { res.push_back(cur); return; }
        dfs(i+1);
        cur.push_back(nums[i]); dfs(i+1); cur.pop_back();
    };
    dfs(0); return res;
}
vector<vector<int>> combinationSum(vector<int> cand, int target) {
    sort(cand.begin(), cand.end());
    vector<vector<int>> res; vector<int> cur;
    function<void(int,int)> dfs = [&](int i, int rem){
        if (rem==0){ res.push_back(cur); return; }
        for (int j=i; j<cand.size() && cand[j]<=rem; ++j) {
            if (j>i && cand[j]==cand[j-1]) continue;
            cur.push_back(cand[j]); dfs(j, rem - cand[j]); cur.pop_back();
        }
    };
    dfs(0, target); return res;
}

/********** 24) PERMUTATIONS (used-array / swap) **********/
vector<vector<int>> permute(vector<int> nums) {
    vector<vector<int>> res; vector<int> cur; vector<int> used(nums.size(), 0);
    function<void()> dfs = [&](){
        if (cur.size() == nums.size()) { res.push_back(cur); return; }
        for (int i=0;i<nums.size();++i) if (!used[i]) {
            used[i]=1; cur.push_back(nums[i]); dfs(); cur.pop_back(); used[i]=0;
        }
    };
    dfs(); return res;
}
void permuteSwapHelper(vector<int>& a, int l, vector<vector<int>>& res) {
    if (l==a.size()) { res.push_back(a); return; }
    unordered_set<int> seen;
    for (int i=l;i<a.size();++i) if (!seen.count(a[i])) {
        seen.insert(a[i]); swap(a[l],a[i]); permuteSwapHelper(a,l+1,res); swap(a[l],a[i]);
    }
}

/********** 25) 1D DP: KNAPSACK / COIN CHANGE **********/
int knapsack01(vector<int> wt, vector<int> val, int W) {
    vector<int> dp(W+1, 0);
    for (int i=0;i<wt.size();++i)
        for (int w=W; w>=wt[i]; --w)
            dp[w] = max(dp[w], dp[w-wt[i]] + val[i]);
    return dp[W];
}
int coinChangeMinCoins(vector<int> coins, int amount) {
    const int INF = 1e9; vector<int> dp(amount+1, INF); dp[0]=0;
    for (int c: coins) for (int x=c; x<=amount; ++x) dp[x] = min(dp[x], dp[x-c]+1);
    return dp[amount] >= INF ? -1 : dp[amount];
}
int coinChangeWays(vector<int> coins, int amount) {
    vector<int> dp(amount+1, 0); dp[0]=1;
    for (int c: coins) for (int x=c; x<=amount; ++x) dp[x] += dp[x-c];
    return dp[amount];
}

/********** 26) GRID DP (paths / min path) **********/
int uniquePathsWithObstacles(vector<vector<int>>& g) {
    int n=g.size(), m=g[0].size(); vector<int> dp(m,0); dp[0]=!g[0][0];
    for (int i=0;i<n;++i) for (int j=0;j<m;++j) {
        if (g[i][j]) dp[j]=0;
        else if (j>0) dp[j]+=dp[j-1];
    }
    return dp[m-1];
}
int minPathSum(vector<vector<int>>& g) {
    int n=g.size(), m=g[0].size(); vector<int> dp(m, INT_MAX); dp[0]=0;
    for (int i=0;i<n;++i) for (int j=0;j<m;++j) {
        dp[j] = min(dp[j], j?dp[j-1]:INT_MAX);
        dp[j] += g[i][j];
        if (j==0 && i==0) dp[j]=g[0][0];
    }
    return dp[m-1];
}

/********** 27) LIS (O(n log n)) **********/
int LIS_Length(const vector<int>& a) {
    vector<int> tail;
    for (int x : a) {
        auto it = lower_bound(tail.begin(), tail.end(), x);
        if (it == tail.end()) tail.push_back(x); else *it = x;
    }
    return (int)tail.size();
}

/********** 28) LCS / EDIT DISTANCE **********/
int LCS_Length(const string& a, const string& b) {
    int n=a.size(), m=b.size(); vector<int> dp(m+1,0);
    for (int i=1;i<=n;++i) {
        int prev=0;
        for (int j=1;j<=m;++j) {
            int cur=dp[j];
            dp[j]= (a[i-1]==b[j-1]) ? prev+1 : max(dp[j], dp[j-1]);
            prev=cur;
        }
    }
    return dp[m];
}
int editDistance(const string& a, const string& b) {
    int n=a.size(), m=b.size(); vector<int> dp(m+1);
    iota(dp.begin(), dp.end(), 0);
    for (int i=1;i<=n;++i) {
        int prev = dp[0]; dp[0]=i;
        for (int j=1;j<=m;++j) {
            int tmp = dp[j];
            if (a[i-1]==b[j-1]) dp[j]=prev;
            else dp[j]=1+min({dp[j], dp[j-1], prev});
            prev = tmp;
        }
    }
    return dp[m];
}

/********** 29) GREEDY: INTERVAL SCHEDULING **********/
int maxNonOverlappingIntervals(vector<vector<int>> iv) {
    sort(iv.begin(), iv.end(), [](auto& x, auto& y){ return x[1]<y[1]; });
    int cnt = 0, end = INT_MIN;
    for (auto& v : iv) if (v[0] >= end) { cnt++; end = v[1]; }
    return cnt;
}

/********** 30) HEAPS: TOP-K / K-WAY MERGE **********/
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int,int> f; for (int x: nums) f[x]++;
    using P = pair<int,int>; priority_queue<P, vector<P>, greater<P>> pq;
    for (auto& [x,c] : f) { pq.push({c,x}); if (pq.size() > k) pq.pop(); }
    vector<int> res; while (!pq.empty()) { res.push_back(pq.top().second); pq.pop(); }
    reverse(res.begin(), res.end()); return res;
}
vector<int> mergeKSorted(const vector<vector<int>>& a) {
    using T = tuple<int,int,int>; // val, row, idx
    priority_queue<T, vector<T>, greater<T>> pq;
    int n=a.size(); for (int i=0;i<n;++i) if (!a[i].empty()) pq.emplace(a[i][0], i, 0);
    vector<int> res;
    while (!pq.empty()) {
        auto [v,i,j]=pq.top(); pq.pop(); res.push_back(v);
        if (j+1 < a[i].size()) pq.emplace(a[i][j+1], i, j+1);
    }
    return res;
}

/********** 31) BIT TRICKS: count bits / subsets / XOR **********/
vector<int> countBits(int n) {
    vector<int> dp(n+1,0);
    for (int i=1;i<=n;++i) dp[i] = dp[i>>1] + (i&1);
    return dp;
}
vector<int> allSubsetsMask(int n) { // iterate masks [0,(1<<n)-1]
    vector<int> masks; masks.reserve(1<<n);
    for (int m=0; m<(1<<n); ++m) masks.push_back(m);
    return masks;
}
int countSubarraysWithXorK(const vector<int>& a, int K) {
    unordered_map<int,int> cnt; cnt[0]=1; int x=0, ans=0;
    for (int v: a) { x ^= v; if (cnt.count(x^K)) ans += cnt[x^K]; cnt[x]++; }
    return ans;
}

/********** 32) MATH BASICS: GCD/LCM / FAST POWER **********/
long long gcdll(long long a, long long b) { return b ? gcdll(b, a%b) : a; }
long long lcmll(long long a, long long b) { return a / gcdll(a,b) * b; }
long long binpow(long long a, long long e) {
    long long r=1; while (e) { if (e&1) r=r*a; a=a*a; e>>=1; } return r;
}
long long modpow(long long a, long long e, long long mod) {
    long long r=1%mod; a%=mod; while (e){ if (e&1) r=r*a%mod; a=a*a%mod; e>>=1; } return r;
}
// ```

// Want this split into multiple files, or with sample problems attached to each template? I can also add a lightweight header-only “patterns.h” with all of the above and a driver scaffold.