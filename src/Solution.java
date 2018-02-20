import org.omg.PortableServer.LIFESPAN_POLICY_ID;

import java.lang.reflect.Array;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Created by GuoJianFeng on 8/29/17.
 */
public class Solution {

    private static class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    public class Interval {
        int start;
        int end;
        Interval() { start = 0; end = 0; }
        Interval(int s, int e) { start = s; end = e; }
    }

    //private static TreeSet<NameAndAmount> treeSet;
//    private static Map<String, NameAndAmount> map;
//
//    static class NameAndAmount {
//        String name;
//        int amount;
//
//        public NameAndAmount(String name, int amount) {
//            this.name = name;
//            this.amount = amount;
//        }
//
//        public NameAndAmount(String name) {
//            this.name = name;
//            this.amount = 0;
//        }
//    }
//
//    public static void update(String company, int amount) {
//        NameAndAmount ele = null;
//        if (map.containsKey(company)) {
//            ele = map.get(company);
//            treeSet.remove(ele);
//            ele.amount += amount;
//        } else {
//            ele = new NameAndAmount(company);
//            ele.amount += amount;
//            map.put(company, ele);
//        }
//        treeSet.add(ele);
//    }
//
//    public static List<String> topK(int k) {
//        List<String> res = new ArrayList<>();
//        int idx = 0;
//        Iterator<NameAndAmount> iter = treeSet.iterator();
//        while (iter.hasNext() && idx++ < k) {
//            NameAndAmount ele = iter.next();
//            res.add(ele.name);
//        }
//        return res;
//    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len1 = nums1.length, len2 = nums2.length;
        int[] longer, shorter;
        if (len1 <= len2) {
            longer = nums2;
            shorter = nums1;
        } else {
            longer = nums1;
            shorter = nums2;
        }
        // n is the longer length, m is the shorter length
        int m = Math.min(len1, len2), n = Math.max(len1, len2);
        int from = 0, to = m;
        while (from <= to) {
            boolean find = false;
            int i = (from + to) / 2, j = (m + n + 1) / 2 - i;
            // since n > m., j cannot be zero;
            //instead using j == 0 , we should use i == m
            if ((i == 0 || shorter[i - 1] <= longer[j]) && (i == m || longer[j - 1] <= shorter[i])) {
                // find the median
                find = true;
            } else if ((i == 0 || shorter[i - 1] <= longer[j]) && (i == m || longer[j - 1] > shorter[i])) {
                // i is too small
                from = i + 1;
            } else if ((i == 0 || shorter[i - 1] > longer[j]) && (i == m || longer[j - 1] <= shorter[i])) {
                // i is too big
                to = i - 1;
            }
            if (find) {
                // i can vary from 0 to m, j vary from 0（when m == n）to n when m == n）
                int maxOfLeft = -1;
                if (i == 0) {
                    maxOfLeft = longer[j - 1];
                } else if (j == 0) {
                    maxOfLeft = shorter[i - 1];
                } else {
                    maxOfLeft = Math.max(shorter[i - 1], longer[j - 1]);
                }
                if ((m + n) % 2 == 1) return maxOfLeft;

                int minOfRight = -1;
                if (i == m) {
                    minOfRight = longer[j];
                } else if (j == n) {
                    minOfRight = shorter[i];
                } else {
                    minOfRight = Math.min(shorter[i], longer[j]);
                }
                return ((double) maxOfLeft + (double) minOfRight) / 2;
            }
        }
        return -1;
    }

    public int lengthOfLongestSubstring(String s) {
        int[] counter = new int[256];
        Arrays.fill(counter, 1);
        int from = 0, to = 0, max = 0;
        while (to < s.length()) {
            boolean dup = true;
            if (counter[s.charAt(to)]-- == 1) {
                dup = false;
            } else {
                dup = true;
            }
            if (dup) {
                max = Math.max(max, to - from);
                while (dup) {
                    if (counter[s.charAt(from++)]++ == 0) {
                        continue;
                    } else {
                        dup = false;
                    }
                }
            }
            to++;
        }
        // corner case when no dup in the whole array
        max = Math.max(max, to - from);
        return max;
    }

    public int maxKilledEnemies(char[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) return 0;

        int row = grid.length, col = grid[0].length;
        int[] rowSum = new int[row], colSum = new int[col];
        int res = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (i == 0 || grid[i - 1][j] == 'W') {
                    // wall down element
                    colSum[j] = 0;
                    for (int k = i; k < row; k++) {
                        if (grid[k][j] == 'E') {
                            colSum[j] += 1;
                        } else if (grid[k][j] == 'W') {
                            break;
                        }
                    }
                }
                if (j == 0 || grid[i][j - 1] == 'W') {
                    rowSum[i] = 0;
                    for (int k = j; k < col; k++) {
                        if (grid[i][k] == 'E') {
                            rowSum[i] += 1;
                        } else if (grid[j][k] == 'W') {
                            break;
                        }
                    }
                }

                if (grid[i][j] == '0') {
                    res = Math.max(res, rowSum[i] + colSum[j]);
                }
            }
        }
        return res;
    }

    public int lengthLongestPath(String input) {
        Stack<Integer> stack = new Stack<>();
        String[] strs = input.split("\n");
        int crtLen = strs[0].length(), max = 0;
        stack.push(strs[0].length());
        for (int i = 1; i < strs.length; i++) {
            String str = strs[i];
            int level = countLevel(str), strLen = str.length();
            while (stack.size() > level) {
                crtLen = crtLen - stack.pop() - 1;
            }
            crtLen += strLen + 1 - level;
            if (str.indexOf('.') != -1 && crtLen > max) {
                max = crtLen;
            }
            stack.push(strLen - level);
        }
        return max;
    }

    private int countLevel(String str) {
        int level = 0, idx = 0;
        while (str.charAt(idx++) == '\t') {
            level += 1;
        }
        return level;
    }

    public ListNode sortList(ListNode head) {
        if(head == null || head.next == null) {
            return head;
        }

        ListNode fast = head.next, slow = head;
        while(fast.next != null) {
            fast = fast.next;
            slow = slow.next;
            if(fast.next != null) {
                fast = fast.next;
            }
        }
        ListNode next = slow.next;
        slow.next = null;
        System.out.println(next.val);
        head = sortList(head);
        next = sortList(next);
        ListNode res = merge(head, next);
        return res;
    }

    private ListNode merge(ListNode l1, ListNode l2) {
        ListNode dommy = new ListNode(-1), p = dommy;
        ListNode p1 = l1, p2 = l2;
        while(p1 != null && p2 != null) {
            if(p1.val < p2.val) {
                p.next = p1;
                p1 = p1.next;
                p = p.next;
            } else {
                p.next = p2;
                p2 = p2.next;
                p = p.next;
            }
        }
        while(p1 != null) {
            p.next = p1;
            p1 = p1.next;
            p = p.next;
        }
        while(p2 != null) {
            p.next = p2;
            p2 = p2.next;
            p = p.next;
        }
        return dommy.next;
    }

    public boolean isMatch(String s, String p) {
        int slen = s.length(), plen = p.length();
        boolean[][] dp = new boolean[plen + 1][slen + 1];
        dp[0][0] = true;
        for(int i = 1; i <= plen; i++) {
            if(p.charAt(i - 1) != '*'){
                continue;
            } else {
                dp[i][0] = dp[i - 2][0];
            }
        }
        for(int i = 1; i <= plen; i++) {
            for(int j = 1; j <= slen; j++) {
                if(p.charAt(i - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if(p.charAt(i - 1) == '*'){
                    // case for 'x*' is empty
                    boolean empty = dp[i - 2][j];
                    // case for single char
                    boolean single = dp[i - 1][j];

                    dp[i][j] = empty || single;
                    // case for multi dup char
                    char prev = p.charAt(i - 2);
                    if(prev == s.charAt(j - 1) || (prev == '.' && (j == 1 || s.charAt(j - 1) == s.charAt(j - 2)))) {
                        dp[i][j] = dp[i][j - 1] || dp[i][j];
                    }
                } else {
                    if(p.charAt(i - 1) == s.charAt(j - 1)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                }
            }
        }
        return dp[plen][slen];
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if(lists == null || lists.length == 0) return null;
        ListNode dommy = new ListNode(-1), pointer = dommy;
        PriorityQueue<ListNode> queue = new PriorityQueue<>((a, b) -> a.val - b.val);
        int len = lists.length;
        ListNode[] p = new ListNode[len];
        for(int i = 0; i < len; i++) {
            if(lists[i] != null) {
                p[i] = lists[i];
                queue.add(lists[i]);
            }
        }
        while(!queue.isEmpty()) {
            ListNode head = queue.poll();
            pointer.next = head;
            pointer = pointer.next;
            if(head.next != null) {
                queue.add(head.next);
            }
        }
        return dommy.next;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        char[] chars = s.toCharArray();
        boolean[] dp = new boolean[chars.length + 1];

        // i j stands for string index, not dp[] index
        for(int i = 0; i < chars.length; i++) {
            for(int j = i + 1; j <= chars.length; j++) {
                String sub = s.substring(i, j);
                if(wordDict.contains(sub) && dp[i]) {
                    dp[j] = true;
                }
            }
        }
        return dp[s.length()];
    }

    int maxProduct(int A[]) {
        int n = A.length;
        // store the result that is the max we have found so far
        int r = A[0];

        // imax/imin stores the max/min product of
        // subarray that ends with the current number A[i]
        for (int i = 1, imax = r, imin = r; i < n; i++) {
            // multiplied by a negative makes big number smaller, small number bigger
            // so we redefine the extremums by swapping them
            if (A[i] < 0) {
                int tmp = imax;
                imax = imin;
                imin = imax;
            }

            // max/min product for the current number is either the current number itself
            // or the max/min by the previous number times the current one
            imax = Math.max(A[i], imax * A[i]);
            imin = Math.min(A[i], imin * A[i]);

            // the newly computed max value is a candidate for our global result
            r = Math.max(r, imax);
        }
        return r;
    }

    public int myAtoi(String str) {
        if(str == null || str.length() == 0) return 0;
        String s = str.trim();
        StringBuilder sb = new StringBuilder();
        boolean sign = false;
        for(int i = 0; i < s.length(); i++) {
            // maybe there is + or -
            char c = s.charAt(i);
            if(c >= '0' && c <= '9') {
                sb.append(c);
            } else {
                if(i == 0) {
                    if(c == '-') {
                        sign = true;
                    } else if(c == '+') {
                        continue;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        String max = String.valueOf(Integer.MAX_VALUE), min = String.valueOf(Integer.MIN_VALUE);
        int len = sb.length();
        String num = sb.toString();
        if(len > 10) {
            if(sign) {
                return Integer.MIN_VALUE;
            } else {
                return Integer.MAX_VALUE;
            }
        } else if(len == 10) {
            if(sign) {
                if(("-" + num).compareTo(min) >= 0) {
                    return Integer.MIN_VALUE;
                } else {
                    try {
                        return -1 * Integer.parseInt(num);
                    } catch (Exception e) {
                        return 0;
                    }
                }
            } else {
                if(num.compareTo(max) >= 0) {
                    return Integer.MAX_VALUE;
                } else {
                    try {
                        return Integer.parseInt(num);
                    } catch (Exception e) {
                        return 0;
                    }
                }
            }
        } else {
            if(sign) {
                try {
                    return -1 * Integer.parseInt(num);
                } catch (Exception e) {
                    return 0;
                }
            } else {
                try {
                    return Integer.parseInt(num);
                } catch (Exception e) {
                    return 0;
                }
            }
        }
    }

    public List<Interval> merge(List<Interval> intervals) {
        if(intervals == null || intervals.size() <= 1) return intervals;

        List<Interval> res = new ArrayList<>();
        Collections.sort(intervals, new Comparator<Interval>() {
            public int compare(Interval a, Interval b) {
                return a.start - b.start;
            }
        });


        Interval last = intervals.get(0);
        for(int i = 1; i < intervals.size(); i++) {
            Interval interval = intervals.get(i);
            if(interval.start > last.end) {
                // non-overlap
                res.add(new Interval(last.start, last.end));
                last = interval;
            } else {
                if(interval.end <= last.end) {
                    continue;
                } else {
                    last.end = interval.end;
                }
            }
        }
        res.add(last);
        return res;
    }

    public String longestPalindrome(String s) {
        int len = s.length();
        int max = 1, idx = 0;
        boolean[][] dp = new boolean[len + 1][len + 1];
        for(int i = 1; i <= len; i++) dp[i][i] = true;

        for(int offset = 1; offset <= len - 1; offset++) {
            for(int i = 1; i + offset <= len; i++) {
                int j = i + offset;
                if(s.charAt(i - 1) == s.charAt(j - 1) && (i + 1 > j - 1 || dp[i + 1][j - 1])) {
                    dp[i][j] = true;
                    if(max < offset + 1) {
                        max = offset + 1;
                        idx = i - 1;

                    }
                }
            }
        }
        return s.substring(idx, idx + max);
    }

    public String reverseString(String s) {
        char[] chars = s.toCharArray();
        for(int start = 0, end = s.length() - 1; start < end; start++, end--) {
            char tmp = chars[start];
            chars[start] = chars[end];
            chars[end] = tmp;
        }
        return new String(chars);
    }

    public int largestRectangleArea(int[] height) {
        int len = height.length;
        Stack<Integer> s = new Stack<Integer>();
        int maxArea = 0;
        for(int i = 0; i <= len; i++){
            int h = (i == len ? 0 : height[i]);
            if(s.isEmpty() || h >= height[s.peek()]){
                s.push(i);
            }else{
                int tp = s.pop();
                maxArea = Math.max(maxArea, height[tp] * (s.isEmpty() ? i : i - 1 - s.peek()));
                i--;
            }
        }
        return maxArea;
    }



    public int longestConsecutive(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        int max = 0;
        for(int i = 0; i < nums.length; i++) {
            int left = map.getOrDefault(nums[i] - 1, 0);
            int right = map.getOrDefault(nums[i] + 1, 0);
            int len = left + right + 1;
            map.put(nums[i] - left, len);
            map.put(nums[i] + right, len);
            max = Math.max(max, len);
        }
        return max;
    }

    public int longestConsecutiveII(int[] nums) throws Exception {
        int res = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int num : nums) {
            if(!map.containsKey(num)) {
                int left = map.containsKey(num - 1) ? map.get(num - 1) : 0;
                int right = map.containsKey(num + 1) ? map.get(num + 1) : 0;
                int tmp = 1 + left + right;
                map.put(num, tmp);
                map.put(num - left, tmp);
                map.put(num + right, tmp);
                res = Math.max(res, tmp);
            }
        }
        return res;
    }

    // when exhasted a tree, we push "#" to denote that this tree is valid
    public boolean isValidSerialization(String preorder) {
        Stack<String> stack = new Stack<>();
        String[] strs = preorder.split(",");

        for(int i = 0; i < strs.length; i++) {
            String str = strs[i];
            while(str.equals("#") && !stack.isEmpty() && stack.peek().equals("#")) {
                stack.pop();
                if(stack.isEmpty()) {
                   return false;
                }
                stack.pop();
            }
            stack.push(str);
        }
        return stack.size() == 1 && stack.peek().equals("#");
    }

    static int maximumDifference(int nodes, int[] a, int[] b) {
        int result = Integer.MIN_VALUE;
        int[] father = new int[nodes + 1];
        for (int i = 0; i < father.length; i++) {
            father[i] = i;
        }
        for (int i = 0; i < a.length; i++) {
            union(a[i], b[i], father);
        }
        for (int i = 1; i < father.length; i++) {
            result = Math.max(result, i - find(i, father));
        }
        return result;
    }

    public static int find(int x, int[] father) {
        if (father[x] == x) {
            return x;
        }
        father[x] = find(father[x], father);
        return father[x];
    }

    public static void union(int a, int b, int[] father) {
        int root_a = find(a, father);
        int root_b = find(b, father);
        if (root_a != root_b) {
            if (root_a > root_b) {
                father[root_a] = root_b;
            } else {
                father[root_b] = root_a;
            }
        }
    }

    public int kEmptySlots(int[] flowers, int k) {
        int[] days =  new int[flowers.length];
        for(int i=0; i<flowers.length; i++)days[flowers[i] - 1] = i + 1;
        int left = 0, right = k + 1, res = Integer.MAX_VALUE;
        for(int i = 0; right < days.length; i++){
            if(days[i] < days[left] || days[i] <= days[right]){
                if(i == right) {
                    res = Math.min(res, Math.max(days[left], days[right]));   //we get a valid subarray
                    System.out.println("left = " + days[left] + "   right = " + days[right]);
                }
                left = i;
                right = k + 1 + i;
            }
        }
        return (res == Integer.MAX_VALUE)? -1 : res;
    }

    public int findDuplicate(int[] nums) {
        if(nums.length > 1) {
            int slow = nums[0];
            int fast = nums[nums[0]];

            while (slow != fast) {
                slow = nums[slow];
                fast = nums[nums[fast]];
            }
            fast = 0;
            while (fast != slow) {
                fast = nums[fast];
                slow = nums[slow];
            }
            return slow;
        }
        return -1;
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    int maxPathSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        helper(root);
        return maxPathSum;
    }

    private int helper(TreeNode root) {
        if(root == null) return 0;

        int left = root.val + helper(root.left);
        int right = root.val + helper(root.right);
        maxPathSum = Math.max(Math.max(root.val, left), Math.max(right, left + right - root.val));
        System.out.println("pathsum = " + maxPathSum);
        System.out.println("left = " + left);
        System.out.println("right = " + right);
        return Math.max(Math.max(root.val, left), right);
    }

    // key format startIndex + : + endIndex
    HashMap<String, List<Integer>> map = new HashMap<>();

    public List<Integer> diffWaysToCompute(String input)  {
        return helper(input, 0, input.length() - 1);
    }

    private List<Integer> helper(String input, int start, int end) {
        String key = start + ":" + end;
        if(map.containsKey(key)) {
            return map.get(key);
        }

        List<Integer> res = new ArrayList<>();
        for(int i = start; i <= end; i++) {
            char c = input.charAt(i);
            if(c == '+' || c == '-' || c == '*') {
                List<Integer> left = helper(input, start, i - 1);
                List<Integer> right = helper(input, i + 1, end);

                for(int num1 : left) {
                    for(int num2 : right) {
                        if(c == '+') {
                            res.add(num1 + num2);
                        } else if(c == '-') {
                            res.add(num1 - num2);
                        } else if(c == '*') {
                            res.add(num1 * num2);
                        }
                    }
                }
            }
        }
        if(res.size() == 0) {
            res.add(Integer.parseInt(input.substring(start, end + 1)));
        }

        map.put(key, res);
        return res;
    }

    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (a, b) ->{
            if(a[1] == b[1]) {
                return a[0] - b[0];
            } else {
                return a[1] - b[1];
            }
        });

        for(int i = 0; i < people.length; i++) {
            int count = 0;
            int[] person = new int[]{people[i][0], people[i][1]};
            for(int j = 0; j < i; j++) {
                if(people[j][0] >= person[0]) {
                    count += 1;
                }
                if(count > person[1]) {
                    // insert at position j
                    for(int k = i; k > j; k--) {
                        people[k] = people[k - 1];
                    }
                    people[j] = person;
                    break;
                }
            }
        }
        return people;
    }

//    public boolean isNumber(String s) {
//        //String patter1 = "[-+]?(([0-9]+(\\.[0-9]*)?)|\\.[0-9]+)(e[-+]?[0-9]+)?";
//        //return Pattern.matches(patter1, s);
//
//    }


    public String minWindow(String s, String t) {
        int[] map = new int[128];
        for(char c : t.toCharArray()) {
            map[c]++;
        }
        int counter = t.length(), from = 0, to = 0, dis = Integer.MAX_VALUE, head = 0;
        while(to < s.length()) {
            if(map[s.charAt(to++)]-- > 0) counter--;
            while(counter == 0) {
                if(to - from < dis) {
                    dis = to - from;
                    head = from;
                }
                if(map[s.charAt(from++) ]++ == 0) {
                    counter++;
                }
            }
        }
        return dis == Integer.MAX_VALUE? "" : s.substring(head, head + dis);
    }

    public int findKthLargest(int[] a, int k) {
        int n = a.length;
        int p = quickSelect(a, 0, n - 1, n - k + 1);
        return a[p];
    }
    // return the index of the kth smallest number
    int quickSelect(int[] a, int lo, int hi, int k) {
        // use quick sort's idea
        // put nums that are <= pivot to the left
        // put nums that are  > pivot to the right
        int i = lo, j = hi, pivot = a[hi];
        while (i < j) {
            if (a[i++] > pivot) swap(a, --i, --j);
        }
        swap(a, i, hi);

        // count the nums that are <= pivot from lo
        int m = i - lo + 1;

        // pivot is the one!
        if (m == k)     return i;
            // pivot is too big, so it must be on the left
        else if (m > k) return quickSelect(a, lo, i - 1, k);
            // pivot is too small, so it must be on the right
        else            return quickSelect(a, i + 1, hi, k - m);
    }

    void swap(int[] a, int i, int j) {
        int tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
    }

    public int longestSubstring(String s, int k) {
        return helper(s, k);
    }

    private int helper(String s, int k) {
        if(s == null || s.length() == 0) return 0;

        int[] counter = new int[26];

        for(int i = 0; i < s.length(); i++) {
            counter[s.charAt(i) - 'a']++;
        }
        for(int i = 0; i < 26; i++) {
            if(counter[i] != 0 && counter[i] < k) {
                int idx = s.indexOf((char)('a' + i));
                return Math.max(helper(s.substring(0, idx), k), helper(s.substring(idx + 1), k));
            }
        }
        return s.length();
    }

    class UndirectedGraphNode {
        int label;
        List<UndirectedGraphNode> neighbors;
        UndirectedGraphNode(int x) {
            label = x;
            neighbors = new ArrayList<UndirectedGraphNode>();
        }
    }

    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        if(node == null) return null;

        HashMap<Integer, UndirectedGraphNode> map = new HashMap<>();
        UndirectedGraphNode newNode = new UndirectedGraphNode(node.label);
        map.put(node.label, newNode);
        Queue<UndirectedGraphNode> queue = new LinkedList<>();
        queue.add(node);
        while(!queue.isEmpty()) {
            UndirectedGraphNode head = queue.poll();
            // copy this node itself
            UndirectedGraphNode copyHead = map.get(head.label);
            // copy the relations of head
            for(UndirectedGraphNode neigh : head.neighbors) {
                // only add new node to avoid duplication
                if(!map.containsKey(neigh.label)) {
                    queue.add(neigh);
                    UndirectedGraphNode neighCopy = new UndirectedGraphNode(neigh.label);
                    map.put(neigh.label, neighCopy);
                }
                copyHead.neighbors.add(map.get(neigh.label));
            }
        }
        return newNode;
    }

    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        int len = 0;
        for(int num : nums) {
            int i = Arrays.binarySearch(dp, 0, len, num);
            if(i < 0) {
                i = -(i + 1);
            }
            dp[i] = num;
            if(len == i) {
                len++;
            }
        }
        return len;
    }

    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        helper(res, k, 1, n, list);
        return res;
    }

    private void helper(List<List<Integer>> res, int remain, int start, int end, List<Integer> prev) {
        if(remain == 0) {
            res.add(new ArrayList<>(prev));
            return;
        }

        for(int i = start; i <= end; i++) {
            prev.add(i);
            helper(res, remain - 1, i + 1, end, prev);
            // resotre
            prev.remove(prev.size() - 1);
        }
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if(k == 1 || head ==null) {
            return head;
        }

        ListNode dommy = new ListNode(-1), fast = dommy, slow = dommy;
        dommy.next =  head;
        int idx = 0;
        while(idx == 0) {
            while(idx++ < k) {
                // move to the group end
                if(fast.next != null) {
                    fast = fast.next;
                } else {
                    // fast.next == null, reach the end
                    return dommy.next;
                }
            }
            // reverse the group
            ListNode prev = slow;
            slow = slow.next;
            slow = reverse(prev, slow, fast);
            fast = slow;
            idx = 0;
        }

        return dommy.next;
    }

    private ListNode reverse(ListNode prev, ListNode start, ListNode end) {
        ListNode startnext = start.next;
        while(start != end) {
            ListNode nextnext = startnext.next;
            startnext.next = start;
            start = startnext;
            startnext = nextnext;
        }
        prev.next.next = startnext;
        ListNode res = prev.next;
        prev.next = end;
        return res;
    }

    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] res = new int[nums.length - k + 1];
        Deque<Integer> queue = new ArrayDeque<>();
        int idx = 0;
        for(int i = 0; i < nums.length; i++) {
            // remove numbers out of range k
            if(!queue.isEmpty() && queue.peek() < i - k + 1) {
                queue.poll();
            }
            // remove smaller numbers in k range as they are useless
            while(!queue.isEmpty() && nums[queue.peekLast()] < nums[i]) {
                queue.pollLast();
            }
            queue.add(i);
            if(i >= k - 1) {
                res[idx++] = nums[queue.peek()];
            }
        }
        return res;
    }

    public int minMeetingRooms(Interval[] intervals) {
        Comparator<Interval> comparator = new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start == o2.start ? o1.end - o2.end :o1.start - o2.start;
            }
        };

        Arrays.sort(intervals, comparator);
        int res = 0;
        Queue<Integer> queue = new PriorityQueue<>();
        for(int i = 0; i < intervals.length; i++) {
            Interval interval = intervals[i];
            while(!queue.isEmpty() && interval.start >= queue.peek()) {
                queue.poll();
            }
            queue.add(interval.end);
            res = Math.max(res, queue.size());
        }
        return res;
    }





    public int findShortestSubArray(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        Map<Integer, Integer> first = new HashMap<>();
        Map<Integer, Integer> last = new HashMap<>();
        int degree = 0;
        // get degree
        for(int i = 0; i < nums.length; i++) {
            int num = nums[i];
            // update degree
            map.put(num, map.getOrDefault(num, 0) + 1);
            degree = Math.max(map.get(num), degree);
            //update first
            if(!first.containsKey(num)) {
                first.put(num, i);
            }
            // update last
            last.put(num, i);
        }
        // find subarray

        // get all target num
        List<Integer> list = new ArrayList<>();
        for(int key : map.keySet()) {
            if(map.get(key) == degree) {
                list.add(key);
            }
        }
        // find first and last appearance
        int min = Integer.MAX_VALUE;
        for(int i = 0; i < list.size(); i++) {
            int target = list.get(i);
            int firstAppear = first.get(target);
            int lastAppear = last.get(target);
            min = Math.min(lastAppear - firstAppear + 1, min);
        }
        return min;
    }

    public int countBinarySubstrings(String s) {
        if(s == null || s.length() <= 1) {
            return 0;
        }

        int res = 0;
        List<Integer> list = new ArrayList<>();
        // first get all 01 and 10
        for(int i = 0; i < s.length() - 1; i++) {
            if((s.charAt(i) == '1' && s.charAt(i + 1) == '0') ||
                    (s.charAt(i) == '0' && s.charAt(i + 1) == '1')){
                list.add(i);
            }
        }
        res += list.size();
        int len = 2;
        Queue<Integer> queue = new LinkedList<>();

        while(list.size() > 0) {
            queue.addAll(list);
            list.clear();

            while(!queue.isEmpty()) {
                int start = queue.poll(), end = start + len - 1;
                if(start - 1 >= 0 && s.charAt(start - 1) == s.charAt(start) &&
                        end + 1 < s.length() && s.charAt(end + 1) == s.charAt(end)) {
                    list.add(start - 1);
                }
            }
            res += list.size();
            len += 2;
        }
        return res;
    }


    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = 0;
        for(int num : nums) {
            sum += num;
        }
        if(sum % k != 0) {
            return false;
        }
        int target = sum / k;
        return helper(target, nums, 0, k, 0);
    }

    private boolean helper(int target, int[] nums, int preSum, int remain, int start) {
        if(remain == 1) {
            return true;
        }
        if(preSum == target) {
            return helper(target, nums, 0, remain - 1, 0);
        }
        boolean res = false;
        for(int i = start; i < nums.length; i++) {
            if(helper(target, nums, preSum + nums[i], remain, i + 1)) {
                return true;
            }
        }
        return false;
    }

    public int nthUglyNumber(int n) {
        if(n == 1) return 1;
        int t2 = 0, t3 = 0, t5 = 0; //pointers for 2, 3, 5
        int[] dp = new int[n];
        dp[0] = 1;
        for(int i  = 1; i < n ; i ++) {
            dp[i] = Math.min(Math.min(dp[t2] * 2, dp[t3] * 3), dp[t5] * 5);
            if(dp[i] == dp[t2] * 2) {
                t2++;
            }
            if(dp[i] == dp[t3] * 3) {
                t3++;
            }
            if(dp[i] == dp[t5] * 5){
                t5++;
            }
        }
        return dp[n - 1];
    }

    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        helper(res, new char[]{'(', ')'}, s, 0, 0);
        return res;
    }

    private void helper(List<String> res, char[] opt, String s, int last_i, int last_j) {
        int counter = 0;
        for(int i = last_i; i < s.length(); i++) {
            if(s.charAt(i) == opt[0]) counter++;
            if(s.charAt(i) == opt[1]) counter--;
            if(counter >= 0) continue;
            for(int j = last_j; j <= i; j++) {
                if(s.charAt(j) == opt[1] && (j == last_j || s.charAt(j - 1) != s.charAt(j))) {
                    helper(res, opt, s.substring(0, j) + s.substring(j + 1), i, j);
                }
            }
            return;
        }

        String reverse = new StringBuilder(s).reverse().toString();
        if(opt[0] == '(') {
            helper(res, new char[]{')', '('}, reverse, 0, 0);
        } else {
            res.add(reverse);
        }
    }


    public List<String> addOperators(String num, int target) {
        List<String> res = new ArrayList<>();
        if(num == null || num.length() == 0) return res;
        addOperators(res, "", num, target, 0, 0, 0);
        return res;
    }

    private void addOperators(List<String> res, String path, String num, long target, int start, long val, long multi) {
        if(start == num.length()) {
            if(val == target) {
                res.add(path);
            }
            return;
        }
        for(int i = start; i < num.length(); i++) {
            //if(num.charAt(start) == '0' && i != start) break;
            long crt = Long.parseLong(num.substring(start, i + 1));
            if(start == 0){
                addOperators(res, path + crt, num, target, i + 1, crt, crt);
            } else {
                addOperators(res, path + "+" + crt, num, target, i + 1, val + crt, crt);
                addOperators(res, path + "-" + crt, num, target, i + 1, val - crt, -crt);
                addOperators(res, path + "*" + crt, num, target, i + 1, val - multi + multi * crt, multi * crt);
            }
            /**addOperators(res, path + crt, num, target, i + 1, crt, crt);
             addOperators(res, path + "+" + crt, num, target, i + 1, val + crt, crt);
             addOperators(res, path + "-" + crt, num, target, i + 1, val - crt, -crt);
             addOperators(res, path + "*" + crt, num, target, i + 1, val - multi + multi * crt, multi * crt); */
        }
    }

    public int[] exclusiveTime(int n, List<String> logs) {
        int[] res = new int[n];
        Stack<Integer> stack = new Stack<>();
        int prev = 0;
        for(String log : logs) {
            String[] parts = log.split(":");
            int start = Integer.parseInt(parts[2]);
            if(!stack.isEmpty()) res[stack.peek()] += start - prev;
            prev = start;
            if(parts[1].equals("start")) {
                stack.push(Integer.parseInt(parts[0]));
            } else {
                res[stack.pop()]++;
            }
        }
        return res;
    }

    public void sortColors(int[] nums) {
        int lo = 0, hi = nums.length - 1;
        int idx = 0;
        while(lo < hi && idx < nums.length) {
            if(idx < hi) {
                if(nums[idx] == 0) {
                    swap(nums, lo++, idx--);
                } else if(nums[idx] == 2) {
                    swap(nums, hi--, idx--);
                }
            }
            idx++;
        }
    }

    public List<int[]> getSkyline(int[][] buildings) {
        List<int[]> res = new ArrayList<>();
        List<int[]> h = new ArrayList<>();

        for(int i = 0; i < buildings.length; i++) {
            // store beginning with the negative value
            h.add(new int[]{buildings[i][0], -buildings[i][2]});
            h.add(new int[]{buildings[i][1], buildings[i][2]});
        }

        Collections.sort(h, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if(o1[0] != o2[0]) {
                    return o1[0] - o2[0];
                } else {
                    return o1[1] - o2[1];
                }
            }
        });

        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> (b - a));
        pq.add(0);
        int prev = 0;
        for(int[] height : h) {
            if(height[1] < 0) {
                pq.add(-height[1]);
            } else {
                pq.remove(height[1]);
            }

            int cur = pq.peek();
            if(cur != prev) {
                res.add(new int[]{height[0], cur});
                prev = cur;
            }
        }
        return res;
    }

    public int combinationSum4(int[] nums, int target) {
        Arrays.sort(nums);
        int[] dp = new int[target + 1];
        Arrays.fill(dp, -1);
        dp[0] = 1;
        return helper(nums, target, dp);
    }

    private int helper(int[] nums, int target, int[] dp) {
        if(dp[target] != -1) {
            return dp[target];
        }

        int res = 0;
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] < target) {
                res += helper(nums, target - nums[i], dp);
                System.out.println("target = " + target + "  adder = " + res);
            } else if(nums[i] == target) {
                res += 1;
                System.out.println("target = " + target + "  adder = " + res);
            } else {
                break;
            }
        }
        dp[target] = res;
        return res;
    }

    public int minSubArrayLen(int s, int[] nums) {
        int start = 0, end = 0, sum = 0, minLen = Integer.MAX_VALUE;

        while(end < nums.length) {
            sum += nums[end++];
            while(sum >= s) {
                minLen = Math.min(minLen, end - start);
                sum -= nums[start++];
            }
        }
        return minLen;
    }

    public int maxSubArrayLen(int[] nums, int k) {
        int sum = 0, max = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        for(int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if(map.containsKey(sum - k)) {
                max = Math.max(i - map.get(sum - k), max);
            }

            if(!map.containsKey(sum)) {
                map.put(sum, i);
            }
        }
        return max;
    }

    public int kEmptySlotsii(int[] flowers, int k) {
        TreeSet<Integer> set = new TreeSet<>();
        Integer lower = null, higher = null;
        for (int i = 0; i < flowers.length; i++) {
            set.add(flowers[i]);
            lower = set.lower(flowers[i]);
            if (lower == null) lower = 0;
            if (flowers[i] - lower  - 1 == k) {
                return i + 1;
            }
            higher = set.higher(flowers[i]);
            if (higher == null) {
                higher = flowers.length + 1;
            }
            if (higher - flowers[i] - 1== k) {
                return i + 1;
            }
        }
        return -1;
    }

    private static String[] LESS_THAN_TWENTY = {"", "One", "Two", "Three", "Four", "Five", "Six",
            "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen",
            "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    private static String[] TENS = {"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty",
            "Seventy", "Eighty", "Ninety"};
    private static String[] THOUSANDS = {"", "Thousand", "Million", "Billion"};



    public String simplifyPath(String path) {
        Stack<String> stack = new Stack<>();
        String[] direct = path.split("/");
        for(int i = 0; i < direct.length; i++) {
            String str = direct[i];
            if(str.equals(".") || str.equals("")) {
                continue;
            } else if(str.equals("..")) {
                if(!stack.isEmpty()) {
                    stack.pop();
                }
            } else {
                stack.push(str);
            }
        }
        String res = "";
        while(!stack.isEmpty()) {
            res = "/" + stack.pop() + res;
        }
        return res.equals("") ? "/" : res;
    }

    public int ladderLength(String beginWord, String endWord, Set<String> wordList) {
        Set<String> beginSet = new HashSet<String>(), endSet = new HashSet<String>();

        int len = 1;
        int strLen = beginWord.length();
        HashSet<String> visited = new HashSet<String>();

        beginSet.add(beginWord);
        endSet.add(endWord);

        while (!beginSet.isEmpty() && !endSet.isEmpty()) {
            if (beginSet.size() > endSet.size()) {
                Set<String> set = beginSet;
                beginSet = endSet;
                endSet = set;
            }

            Set<String> temp = new HashSet<String>();
            for (String word : beginSet) {
                char[] chs = word.toCharArray();

                for (int i = 0; i < chs.length; i++) {
                    for (char c = 'a'; c <= 'z'; c++) {
                        char old = chs[i];
                        chs[i] = c;
                        String target = String.valueOf(chs);

                        if (endSet.contains(target)) {
                            return len + 1;
                        }

                        if (!visited.contains(target) && wordList.contains(target)) {
                            temp.add(target);
                            visited.add(target);
                        }
                        chs[i] = old;
                    }
                }
            }

            beginSet = temp;
            len++;
        }

        return 0;
    }

    public String alienOrder(String[] words) {
        if(words == null || words.length == 0) return "";

        // initialization
        Map<Character, Set<Character>> map = new HashMap<>(); // outer edges
        Map<Character, Integer> degree = new HashMap<>(); // in-degree

        // find char set
        for(String str : words) {
            for(char c : str.toCharArray()) {
                map.put(c, new HashSet<>());
                degree.put(c, 0);
            }
        }

        // construct our graph
        for(int i = 0; i < words.length - 1; i++) {
            String cur = words[i];
            String next = words[i + 1];
            int len = Math.min(cur.length(), next.length());
            for(int j = 0; j < len; j++) {
                char ccur = cur.charAt(j), cnext = next.charAt(j);
                if(ccur != cnext) {
                    // critical, avoid a->b, a->b duplication count
                    if(!map.get(ccur).contains(cnext)) {
                        map.get(ccur).add(cnext);
                        degree.put(cnext, degree.get(cnext) + 1);
                    }
                    break;
                }
            }
        }
        // traveral our graph
        Queue<Character> queue = new LinkedList<>();
        for(char c : degree.keySet()) {
            if(degree.get(c) == 0) queue.add(c);
        }
        StringBuilder sb = new StringBuilder();
        while(!queue.isEmpty()) {
            char c = queue.poll();
            sb.append(c);
            for(char c2: map.get(c)){
                int c2in = degree.get(c2);
                if(c2in == 1) {
                    queue.add(c2);
                    degree.put(c2, c2in - 1);
                } else if(c2in > 1) {
                    degree.put(c2, c2in - 1);
                } else {
                    return "";
                }
            }
        }
        String res = sb.toString();
        if(res.length() == degree.size()) {
            return res;
        } else {
            return "";
        }

    }

    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        boolean bol = true;
        int len = 0;
        StringBuilder line = new StringBuilder();
        // build string without justification
        int i = 0;
        while (i < words.length) {
            while (i < words.length && len <= maxWidth) {
                if(bol) {
                    len += words[i].length();
                    line.append(words[i++]);
                    bol = false;
                    // corner case
                    if(i == words.length) {
                        while(line.length() < maxWidth) {
                            line.append(" ");
                        }
                        res.add(line.toString());
                    }
                } else {
                    if(len + words[i].length() + 1 <= maxWidth) {
                        // can still fill in this word
                        len += words[i].length() + 1;
                        line.append(' ').append(words[i++]);
                        // corner case
                        if(i == words.length) {
                            while(line.length() < maxWidth) {
                                line.append(" ");
                            }
                            res.add(line.toString());
                        }
                    } else {
                        res.add(line.toString());
                        bol = true;
                        len = 0;
                        line = new StringBuilder();
                    }
                }
            }
        }

        // justify
        List<String> rst = new ArrayList<>();
        for(int j = 0; j < res.size() - 1 ; j++) {
            String just = justify(res.get(j), maxWidth);
            rst.add(just);
        }
        rst.add(res.get(res.size() - 1));
        return rst;
    }

    private String justify(String str, int len) {
        String[] words = str.split(" ");
        if(words.length == 1) {
            StringBuilder sb = new StringBuilder(str);
            while (sb.length() < len) sb.append(' ');
            return sb.toString();
        } else {
            int extraSpace = len - str.length(), aver = extraSpace /(words.length - 1);
            int left = extraSpace - aver * (words.length - 1);
            StringBuilder sb = new StringBuilder(words[0]);
            for(int i = 1; i < words.length; i++) {
                int tmp = aver;
                while(tmp-- >= 0) {
                    sb.append(' ');
                }
                if(left -- > 0) {
                    sb.append(' ');
                }
                sb.append(words[i]);
            }
            return sb.toString();
        }
    }

    public ListNode deleteDuplicates(ListNode a) {
        if(a == null || a.next == null) return a;

        ListNode p = a;
        while(p != null) {
            if(p.next != null) {
                if(p.next.val == p.val) {
                    p.next = p.next.next;
                } else {
                    p = p.next;
                }
            } else {
                p = p.next;
            }
        }
        return a;
    }

    public int findNumberOfLIS(int[] nums) {
        int n = nums.length, res = 0, max_len = 0;
        int[] len =  new int[n], cnt = new int[n];
        for(int i = 0; i<n; i++){
            len[i] = cnt[i] = 1;
            for(int j = 0; j <i ; j++){
                if(nums[i] > nums[j]){
                    if(len[i] == len[j] + 1)cnt[i] += cnt[j];
                    if(len[i] < len[j] + 1){
                        len[i] = len[j] + 1;
                        cnt[i] = cnt[j];
                    }
                }
            }
            if(max_len == len[i])res += cnt[i];
            if(max_len < len[i]){
                max_len = len[i];
                res = cnt[i];
            }
        }
        return res;
    }

    public int minCostII(int[][] costs) {
        if(costs == null || costs.length == 0 || costs[0].length == 0) {
            return 0;
        }
        int idx1 = -1, idx2 = -1, n = costs.length, k = costs[0].length;
        for(int i = 0; i < n; i++) {
            int last1 = idx1, last2 = idx2;
            idx1 = -1; idx2 = -1;
            for(int j = 0; j < k; j++) {
                if(j != last1) {
                    costs[i][j] += last1 < 0 ? 0 : costs[i - 1][last1];
                } else {
                    costs[i][j] += last1 < 0 ? 0 : costs[i - 1][last2];
                }

                if(idx1 < 0 || costs[i][j] < costs[i][idx1]) {
                    idx2 = idx1;
                    idx1 = j;
                } else if(idx2 < 0 || costs[i][j] < costs[i][idx2]) {
                    idx2 = j;
                }
            }
        }
        return costs[n - 1][idx1];
    }

    private static final int M = 1000000000 + 7;
    public int numDecodings(String s) {
        int len = s.length();
        long[] dp = new long[len + 1];
        dp[len] = 1;
        for(int i = len - 1; i >= 0; i--) {
            char c = s.charAt(i);
            if(i == len - 1) {
                // in this case, we do not need to consider the following char
                if(c == '0') {
                    dp[i] = 0;
                } else if(c >= '0' && c <= '9') {
                    dp[i] = 1;
                } else {
                    // c == '*'
                    dp[i] = 9;
                }
            } else {
                // when we are not at final position, we need to consider the following character
                if(c == '0') {
                    dp[i] = 0;
                } else if( c >= '1' && c <= '9') {
                    // treat c as a single char
                    dp[i] = (dp[i] + dp[i + 1]) % M;
                    // treat c with its following
                    char cf = s.charAt(i + 1);
                    if(cf >= '0' && cf <= '9') {
                        int num = (c - '0') * 10 + cf - '0';
                        if(num >= 10 && num <= 26) {
                            dp[i] = (dp[i] + dp[i + 2]) % M;
                        }
                    } else {
                        // cf == *
                        if(c == 1) {
                            // 9 possible combination
                            dp[i] = (dp[i] + 9 * dp[i + 2]) % M;
                        } else if(c == 2) {
                            dp[i] = (dp[i] + 6 * dp[i + 2]) % M;
                        }
                    }
                } else {
                    // c == '*'
                    // first, treat c as single char
                    dp[i] = (dp[i] + dp[i + 1] * 9) % M;
                    //// treat c with its following
                    char cf = s.charAt(i + 1);
                    if(cf >= '0' && cf <= '9')  {
                        // * can be 1, and all 0 through 9 are fit
                        dp[i] = (dp[i] + dp[i + 2]) % M;
                        // * can be 2, and only 0 through 6 are valid
                        if(cf >= '0' && cf <= '6') {
                            dp[i] = (dp[i] + dp[i + 2]) % M;
                        }
                    } else {
                        // cf = *
                        dp[i] = (dp[i] + dp[i + 2] * 15) % M;
                    }
                }
            }
        }
        return (int)dp[0];
    }

    public List<String> letterCombinations(String digits) {
        LinkedList<String> ans = new LinkedList<String>();
        if (digits.length()==0) return ans;
        String[] mapping = new String[] {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for(int i =0; i<digits.length();i++){
            int x = Character.getNumericValue(digits.charAt(i));
            while(ans.peek().length()==i){
                String t = ans.remove();
                for(char s : mapping[x].toCharArray())
                    ans.add(t+s);
            }
        }
        return ans;
    }

    int foo(int[][] matrix) {
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0) return 0;

        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m][n];

        for(int i = 1; i < m; i++) {
            if(matrix[i][0] != -1) {
                dp[i][0] = dp[i - 1][0] + matrix[i][0];
            } else {
                break;
            }
        }

        for(int j = 1; j < n; j++) {
            if(matrix[0][j] != -1) {
                dp[0][j] = dp[0][j - 1] +  + matrix[0][j];
            } else {
                break;
            }
        }

        for(int i = 1; i < m; i++) {
            for(int j = 1; j < n; j++) {
                if(matrix[i][j] != -1) {
                    dp[i][j] = matrix[i][j] + Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        int[][] dp2 = new int[m][n];
        // last col
        for(int i = m - 2; i >= 0; i--) {
            if(matrix[i][n - 1] != -1) {
                dp2[i][n - 1] = dp2[i + 1][n - 1] + matrix[i][n - 1];
            } else {
                break;
            }
        }
        // last row
        for(int j = n - 2; j >= 0; j--) {
            if(matrix[m - 1][j] != -1) {
                dp2[m - 1][j] = dp2[m - 1][j + 1] + matrix[m - 1][j];
            } else {
                break;
            }
        }
        for(int i = m - 2; i >= 0; i--) {
            for(int j = n - 2; j >= 0; j--) {
                if(matrix[i][j] != -1) {
                    dp2[i][j] = matrix[i][j] + Math.max(dp2[i][j + 1], dp2[i + 1][j]);
                }
            }
        }
        return dp[m - 1][n - 1] + dp2[0][0];
    }

    public int[] maxSumOfThreeSubarrays(int[] nums, int k) {
        int n = nums.length, maxsum = 0;
        int[] sum = new int[n + 1], posLeft = new int[n], posRight = new int[n], ans = new int[3];
        for (int i = 0; i < n; i++) sum[i + 1] = sum[i] + nums[i];
        // DP for starting index of the left max sum interval
        for (int i = k, tot = sum[k] - sum[0]; i < n; i++) {
            if (sum[i + 1] - sum[i + 1 - k] > tot) {
                posLeft[i] = i + 1 - k;
                tot = sum[i + 1] - sum[i + 1 - k];
            } else
                posLeft[i] = posLeft[i - 1];
        }
        // DP for starting index of the right max sum interval
        // caution: the condition is ">= tot" for right interval, and "> tot" for left interval
        posRight[n - k] = n - k;
        for (int i = n - k - 1, tot = sum[n] - sum[n - k]; i >= 0; i--) {
            if (sum[i + k] - sum[i] >= tot) {
                posRight[i] = i;
                tot = sum[i + k] - sum[i];
            } else
                posRight[i] = posRight[i + 1];
        }
        // test all possible middle interval
        for (int i = k; i <= n - 2 * k; i++) {
            int l = posLeft[i - 1], r = posRight[i + k];
            int tot = (sum[i + k] - sum[i]) + (sum[l + k] - sum[l]) + (sum[r + k] - sum[r]);
            if (tot > maxsum) {
                maxsum = tot;
                ans[0] = l;
                ans[1] = i;
                ans[2] = r;
            }
        }
        return ans;
    }

    public int[] maxSumOfThreeSubarrays2(int[] nums, int k) {
        int n = nums.length, sum = 0;
        int[] preSum = new int[n + 1];
        for(int i = 0; i < n; i++) {
            sum += nums[i];
            preSum[i + 1] = sum;
        }

        int[] left = new int[n];
        int kSum = 0;
        for(int i = k - 1; i < n; i++) {
            int cur = preSum[i + 1] - preSum[i + 1 - k];
            if(cur > kSum) {
                left[i] = i - k + 1;
                kSum = cur;
            } else {
                left[i] = left[i - 1];
            }
        }

        int[] right = new int[n];
        kSum = 0;
        for(int i = n - k; i >= 0; i--) {
            int cur = preSum[i + k] - preSum[i];
            if(cur >= kSum) {
                right[i] = i;
                kSum = cur;
            } else {
                right[i] = right[i + 1];
            }
        }

        kSum = 0;
        int[] res = new int[3];
        for(int i = k; i <= n - 2*k; i++) {
            int l = left[i - 1], r = right[i + k];
            int cur = (preSum[i + k] - preSum[i]) + (preSum[l + k] - preSum[l]) + (preSum[r + k] - preSum[r]);
            if(cur > kSum) {
                kSum = cur;
                res[0] = l - k + 1;
                res[1] = i;
                res[2] = i + k;
            }
        }
        return res;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        // corner case handling
        if(nums == null || nums.length < 3) return res;

        Arrays.sort(nums);
        for(int i = 0; i < nums.length; i++) {
            int first = nums[i];
            // find two nums add up to -first
            int lo = i + 1, hi = nums.length - 1;
            while(lo < hi) {
                int sum = nums[lo] + nums[hi];
                if(sum < -first) {
                    while(lo + 1 < nums.length && nums[lo + 1] == nums[lo]) lo++;
                    lo++;
                } else if(sum > -first) {
                    while(hi - 1 >= 0 && nums[hi - 1] == nums[hi]) hi--;
                    hi--;
                } else {
                    // find our target
                    res.add(Arrays.asList(first, nums[lo], nums[hi]));
                    while(lo + 1 < nums.length && nums[lo + 1] == nums[lo]) lo++;
                    lo++;
                    while(hi - 1 >= 0 && nums[hi - 1] == nums[hi]) hi--;
                    hi--;
                }
            }
            while(i + 1 < nums.length && nums[i] == nums[i + 1]) i++;
        }
        return res;
    }

    public int findTargetSumWays(int[] nums, int S) {
        int res = findTargetSumWays(nums, 0, S);
        return res;
    }

    private int findTargetSumWays(int[] nums, int start, int S) {
        if(start == nums.length - 1) {
            int res = 0;
            if(nums[start] == S) {
                res++;
            }
            if(nums[start] + S == 0){
                res++;
            }
            return res;
        }
        int positive = findTargetSumWays(nums, start + 1, S - nums[start]);
        int negative = findTargetSumWays(nums, start + 1, S + nums[start]);
        return positive + negative;
    }

    public int maximumSwap(int num) {
        char[] chars = String.valueOf(num).toCharArray();
        int len = chars.length;
        int[] idxes = new int[len];
        int max = -1, maxIdx = len - 1;
        for(int i = len - 1; i >= 0; i--) {
            if(i == len - 1) {
                max = chars[i] - '0';
                idxes[i] = i;
            } else {
                int digit = chars[i] - '0';
                if(digit >= max) {
                    max = digit;
                    idxes[i] = i;
                } else {
                    idxes[i] = maxIdx;
                }
            }
        }
        for(int i = 0; i < len; i++) {
            if(idxes[i] != i) {
                // swap i and idxes[i]
                char tmp = chars[i];
                chars[i] = chars[idxes[i]];
                chars[idxes[i]] = tmp;
                break;
            }
        }
        return Integer.parseInt(new String(chars));
    }

    // there can be million of tasks, with just a small cooldown
    public int taskScheduler(int[] tasks, int cooldown) {
        int res = 0;
        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0; i < tasks.length; i++) {
            int task = tasks[i];
            if(queue.contains(task)) {
                // task is still in cd
                res++;
                queue.poll();
                i--;
            } else {
                // queue is not in cd
                res++;
                queue.add(task);
                if(queue.size() == cooldown + 1) {
                    queue.poll();
                }
            }
        }
        return res;
    }

    public float averageFastest(int[] everyFiceSeconds) {
        float meter = 0;
        float second = 0;
        float max = 0;
        Queue<Float> queue = new LinkedList<>();
        for(int i = 0; i < everyFiceSeconds.length; i++) {
            if(everyFiceSeconds[i] == 0) {
                second += 5;
                for(int j = 0; j < 5; j++) {
                    queue.add(Float.MAX_VALUE);
                }
                continue;
            }

           float timePerMeter =  5 / everyFiceSeconds[i];
           for(int j = 0; j < 5; j++) {
               meter += 1;
               second += timePerMeter;
               queue.add(timePerMeter);
               if(meter == 1000) {
                   max = Math.max(max, meter / second);
               } else if(meter > 1000) {
                   meter = meter - 1;
                   float head = queue.poll();
                   if(head != Float.MAX_VALUE) {
                       second = second - head;
                   }
                   max = Math.max(max, meter / second);
               }
           }
        }
        return max;
    }

    private static void swap(Integer a, Integer b){
        Integer tmp = a;
        a = b;
        b = tmp;
    }

    public int topkSum(int[] nums, int k) {
        int[] counter = new int[10];
        for(int num : nums) {
            counter[num - 0]++;
        }
        int kSum = 0, remain = k;
        for(int i = 9; i >= 0 && remain > 0; i--) {
            if(remain >= counter[i]) {
                kSum += counter[i] * i;
                remain -= counter[i];
            } else {
                kSum += remain * i;
                break;
            }
        }
        return kSum;
    }

    private ListNode reverseLinkedList(ListNode head) {
        if(head == null || head.next == null
                || head.next == head || head.next.next == head) {
            return head;
        }
        ListNode p = head, tail = p.next, next = p.next;
        // find tail first
        while(tail != null && tail != head) {
            tail = tail.next;
        }
        // tail can be null or head
        while(next != tail) {
            ListNode nn = next.next;
            next.next = p;
            p = next;
            next = nn;
        }
        if(tail == head) {
            head.next = p;
        } else {
            head.next = null;
        }
        return p;
    }

    public int pivotIndex(int[] nums) {
        if(nums == null || nums.length == 0) return -1;
        int len = nums.length;
        int[] preSum = new int[len + 1];
        int sum = 0;
        for(int i = 0; i < len; i++) {
            preSum[i] = sum;
            sum += nums[i];
        }
        preSum[len] = sum;

        for(int i = 0; i < len; i++) {
            if(preSum[i] == sum - preSum[i + 1]) {
                return i;
            }
        }
        return -1;
    }

    public ListNode[] splitListToParts(ListNode root, int k) {
        ListNode[] res = new ListNode[k];
        if(root == null) {
            return res;
        }

        int len = 0;
        ListNode p = root;
        while(p != null) {
            p = p.next;
            len++;
        }

        int partsLen = len / k, remain = len % k, idx = 0 ;
        if(partsLen == 0) {
            partsLen = 1;
            remain = 0;
            k = len;
        }
        p = root;
        while(idx < k) {
            // move k forward
            res[idx++] = p;
            int i = 0;
            while(++i < partsLen) {
                p = p.next;
            }
            if(remain-- > 0) {
                p = p.next;
            }
            ListNode next = p.next;
            p.next = null;
            p = next;
        }
        return res;
    }

    public String fractionToDecimal(int numerator, int denominator) {
        StringBuilder sb = new StringBuilder();
        if((numerator > 0) ^ (denominator > 0)) sb.append('-');

        long num = (long)numerator, den = (long)denominator;
        num = num % den;
        sb.append(num / den);
        if(num == 0) {
            return sb.toString();
        } else {
            sb.append('.');
        }

        Map<Long, Integer> numToIdx = new HashMap<>();

//        while(num != 0) {
//            long digit = num * 10 / den;
//            num = num * 10 % den;
//            sb.append(digit);
//            if(numToIdx.containsKey(num)) {
//                sb.insert(numToIdx.get(num), "(");
//                sb.append(')');
//                break;
//            } else {
//                numToIdx.put(num, sb.length());
//            }
//        }
        while (num != 0) {
            num *= 10;
            sb.append(num / den);
            num %= den;
            if (numToIdx.containsKey(num)) {
                //int index = map.get(num);
                sb.insert(numToIdx.get(num), "(");
                sb.append(")");
                break;
            }
            else {
                numToIdx.put(num, sb.length());
            }
        }
        return sb.toString();

    }

    public boolean isNumber(String s) {
        s = s.trim();

        boolean pointSeen = false;
        boolean eSeen = false;
        boolean numberSeen = false;
        boolean numberAfterE = true;
        for(int i=0; i<s.length(); i++) {
            if('0' <= s.charAt(i) && s.charAt(i) <= '9') {
                numberSeen = true;
                numberAfterE = true;
            } else if(s.charAt(i) == '.') {
                if(eSeen || pointSeen) {
                    return false;
                }
                pointSeen = true;
            } else if(s.charAt(i) == 'e') {
                if(eSeen || !numberSeen) {
                    return false;
                }
                numberAfterE = false;
                eSeen = true;
            } else if(s.charAt(i) == '-' || s.charAt(i) == '+') {
                if(i != 0 && s.charAt(i-1) != 'e') {
                    return false;
                }
            } else {
                return false;
            }
        }

        return numberSeen && numberAfterE;
    }

    public List<String> wordBreakII(String s, List<String> wordDict) {
        Map<Integer, List<Integer>> idxToPrev = new HashMap<>();
        idxToPrev.put(0, new ArrayList<>());

        for(int i = 1; i <= s.length(); i++) {
            for(int j = 0; j < i; j++) {
                // sub start from j inclusive, end at i exclusive
                String sub = s.substring(j, i);
                if(wordDict.contains(sub) && idxToPrev.containsKey(j)) {
                    if(!idxToPrev.containsKey(i)) {
                        idxToPrev.put(i, new ArrayList<>());
                    }
                    idxToPrev.get(i).add(j);
                }
            }
        }
        List<String> res = new ArrayList<>();
        findPaths(res, idxToPrev, s.length(), new StringBuilder(s));
        return res;
    }

    private void findPaths(List<String> paths, Map<Integer, List<Integer>> map, int end, StringBuilder sb) {
        List<Integer> list = map.get(end);
        for(int i = 0; i < list.size(); i++) {
            StringBuilder next = new StringBuilder(sb);
            int prevIdx = list.get(i);
            if(prevIdx == 0) {
                paths.add(next.toString());
            } else {
                next.insert(prevIdx, " ");
                findPaths(paths, map, prevIdx, next);
            }
        }
    }

    private void inorderIter(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        // to left most node
        while(p != null) {
            stack.push(p);
            p = p.left;
        }
        while(!stack.isEmpty()) {
            p = stack.pop();
            System.out.print(p.val + "  ");
            if (p.right != null) {
                p = p.right;
                while (p != null) {
                    stack.push(p);
                    p = p.left;
                }
            }
        }
    }

    private void preorderIter(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()) {
            TreeNode p = stack.pop();
            System.out.print(p.val + "\t");
            if(p.right != null) {
                stack.push(p.right);
            }
            if(p.left != null) {
                stack.push(p.left);
            }
        }
    }

    private void postOrderIterTwoStack(TreeNode root) {
        Stack<TreeNode> stack1 = new Stack<>();
        Stack<TreeNode> stack2 = new Stack<>();

        stack1.push(root);
        while(!stack1.isEmpty()) {
            TreeNode top = stack1.pop();
            stack2.push(top);
            if(top.left != null) {
                stack1.push(top.left);
            }
            if(top.right != null) {
                stack1.push(top.right);
            }
        }
        while(!stack2.isEmpty()) {
            System.out.print(stack2.pop().val + "   ");
        }
    }

    private void postOrderIterOneStack(TreeNode root) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.push(root);
        TreeNode prev = null;
        while(!stack.isEmpty()) {
            TreeNode cur = stack.peek();
            if(prev == null || prev.left == cur || prev.right == cur) {
                if(cur.left != null) {
                    stack.push(cur.left);
                } else if(cur.right != null) {
                    stack.push(cur.right);
                } else {
                    stack.pop();
                    System.out.print(cur.val + "  ");
                }
            } else if (cur.left == prev) {
                if (cur.right != null) {
                    stack.push(cur.right);
                }
                else {
                    stack.pop();
                    System.out.print(cur.val + "  ");
                }
            } else if (cur.right == prev){
                stack.pop();
                System.out.print(cur.val + "  ");
            }
            prev = cur;
        }
    }

    class Wrapper{
        String elements;
        int count;
        public Wrapper(String elements, int count) {
            this.elements = elements;
            this.count = count;
        }
    }

    private int parseNum(String s, int pos) {
        StringBuilder sb = new StringBuilder();
        for(int i = pos; i < s.length(); i++) {
            char c = s.charAt(i);
            if(c >= '0' && c <= '9') {
                sb.append(c);
            } else {
                break;
            }
        }
        return Integer.parseInt(sb.toString());
    }

    public String minWindowII(String s, String t) {
        int slen = s.length(), tlen = t.length();
        int[][] dp = new int[tlen + 1][slen + 1];
        for(int j = 0; j <= slen; j++) {
            dp[0][j] = j;
        }
        for(int i = 1; i <= tlen; i++) {
            dp[i][0] = -1;
        }
        int start = -1, len = Integer.MAX_VALUE;
        for(int i = 1; i <= tlen; i++) {
            for(int j = 1; j <= slen; j++) {
                if(t.charAt(i - 1) == s.charAt(j - 1) && dp[i - 1][j - 1] != -1) {
                    dp[i][j] = dp[i - 1][j - 1];
                    if(i == tlen) {
                        int newLen = j - dp[i][j];
                        if(newLen < len) {
                            len = newLen;
                            start = dp[i][j];
                        }
                    }
                } else {
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }
        return start == -1 ?  "" : s.substring(start, start + len);
    }

    public List<Integer> selfDividingNumbers(int left, int right) {
        List<Integer> res = new ArrayList<>();
        for(int i = left; i <= right; i++) {
            if(isSelfDividing(i)) {
                res.add(i);
            }
        }
        return res;
    }

    private boolean isSelfDividing(int num) {
        if(num >= 1 && num <= 9) {
            return true;
        }

        int tmp = num;
        while(tmp != 0) {
            int lastDigit = tmp % 10;
            tmp = tmp / 10;
            if(lastDigit == 0 || num % lastDigit != 0) {
                return false;
            }
        }
        return true;
    }

    final int m = 1000000007;
    public int countPalindromicSubsequences(String str) {
        int N = str.length();
        // create a 2D array to store the count
        // of palindromic subsequence
        int[][] cps = new int[N+1][N+1];
        // palindromic subsequence of length 1
        for (int i = 0; i < N; i++) {
            cps[i][i] = 1;
        }
        // check subsequence of length L is
        // palindrome or not
        for (int L = 2; L <= N; L++) {
            for (int i = 0; i < N; i++) {
                // for interval [i,k]
                int k = L + i - 1;
                if (k < N){
                    if (str.charAt(i) == str.charAt(k)) {
                        if(!str.substring(i, k).equals(str.substring(i + 1, k + 1))) {
                            cps[i][k] = (cps[i][k-1] + cps[i+1][k] + 1) % m;
                            System.out.println("dp[" + i + "][" + k + "] = " + cps[i][k]);
                        } else {
                            cps[i][k] = (cps[i][k-1] + 1) % m;
                            System.out.println("dp[" + i + "][" + k + "] = " + cps[i][k]);
                        }
                    } else {
                        if(!str.substring(i, k).equals(str.substring(i + 1, k + 1))) {
                            cps[i][k] = (cps[i][k-1] + cps[i+1][k] - cps[i+1][k-1]) % m;
                            System.out.println("dp[" + i + "][" + k + "] = " + cps[i][k]);
                        } else {
                            cps[i][k] = (cps[i][k-1] - cps[i+1][k-1]) % m;
                            System.out.println("dp[" + i + "][" + k + "] = " + cps[i][k]);
                        }
                    }
                }
            }
        }
        // return total palindromic subsequence
        return cps[0][N-1];
    }

    private final int[][] dir = new int[][]{{0,1}, {0,-1}, {1,0}, {-1,0}};
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int oldColor = image[sr][sc];
        // corner case
        if(oldColor == newColor) {
            return image;
        }
        int m = image.length, n = image[0].length;
        Queue<int[]> queue = new LinkedList<>();
        queue.add(new int[]{sr, sc});
        while(!queue.isEmpty()) {
            int[] pos = queue.poll();
            image[pos[0]][pos[1]] = newColor;
            for(int i = 0; i < 4; i++) {
                int x = pos[0] + dir[i][0];
                int y = pos[1] + dir[i][1];
                if(x >= 0 && x < m && y >= 0 && y < n && image[x][y] == oldColor) {
                    queue.add(new int[]{x, y});
                }
            }
        }
        return image;
    }

    public boolean areSentencesSimilar(String[] words1, String[] words2, String[][] pairs) {
        if(words1.length != words2.length) {
            return false;
        }

        int len = words1.length;
        Map<String, List<String>> map = new HashMap<>();
        for(String[] pair : pairs) {
            if(!map.containsKey(pair[0])) {
                map.put(pair[0], new ArrayList<>());
            }
            if(!map.containsKey(pair[1])) {
                map.put(pair[1], new ArrayList<>());
            }
            // add itself
            map.get(pair[0]).add(pair[0]);
            map.get(pair[1]).add(pair[1]);
            // add the counterpart
            map.get(pair[0]).add(pair[1]);
            map.get(pair[1]).add(pair[0]);
        }
        for(String word : words1) {
            if(!map.containsKey(word)) {
                map.put(word, new ArrayList<>());
                map.get(word).add(word);
            }
        }
        for(int i = 0; i < len; i++) {
            String w1 = words1[i], w2 = words2[i];
            List<String> w1Similar = map.get(w1);
            if(!w1Similar.contains(w2)) {
                return false;
            }
        }
        return true;

    }

    public int[] asteroidCollision(int[] asteroids) {
        if(asteroids == null || asteroids.length == 0) {
            return asteroids;
        }

        Stack<Integer> stack = new Stack<>();
        int zeros = 0;
        for(int i = asteroids.length - 1; i >= 0; i--) {
            if(asteroids[i] < 0) {
                stack.push(i);
            } else {
                // asteroids[i] > 0
                if(stack.isEmpty()) {
                    continue;
                } else {
                    while (!stack.isEmpty()) {
                        if(asteroids[i] > -asteroids[stack.peek()]) {
                            int idx = stack.pop();
                            asteroids[idx] = 0;
                            zeros++;
                        } else if(asteroids[i] == -asteroids[stack.peek()]) {
                            int idx = stack.pop();
                            asteroids[idx] = 0;
                            asteroids[i] = 0;
                            zeros += 2;
                            break;
                        } else {
                            asteroids[i] = 0;
                            zeros++;
                            break;
                        }
                    }
                }
            }
        }
        int resLen = asteroids.length - zeros;
        int[] res = new int[resLen];
        int idx = 0;
        for(int asteroid : asteroids) {
            if(asteroid != 0) {
                res[idx++] = asteroid;
            }
        }
        return res;
    }

//    public int evaluate(String expression) {
//        expression = expression.substring(1, expression.length() - 1);
//        String[] info = expression.split(" ");
//    }
//
//    // this equation do not contains ( or )
//    private int evaluate(String expression, Map<String, Integer> paras) {
//        String oper = expression.substring(0, 3);
//        if(oper.equals("add")) {
//
//        } else if(oper.equals("mul")) {
//
//        } else {
//            // let operation
//        }
//    }


    public String countOfAtoms(String formula) {
        Stack<Integer> times = new Stack<>();
        Stack<TreeMap<String, Integer>> elementsToTimes = new Stack<>();
        TreeMap<String, Integer> res = new TreeMap<>();
        int k = 0, len = formula.length(), pow = 1;
        for(int i = len - 1; i >= 0; i--) {
            char c = formula.charAt(i);
            if(c >= '0' && c <= '9') {
                // do not forget we build in reverse order
                k = (c - '0') * pow + k;
                pow *= 10;
            } else if(c == ')') {
                // start of another formula, do following
                // 1. push k and reset k
                times.push(k);
                k = 0;
                pow = 1;
                elementsToTimes.push(new TreeMap<>(res));
                res = new TreeMap<>();
            } else if(c == '(') {
                // end of patial formula
                TreeMap<String, Integer> tmp = new TreeMap<>(res);
                res = elementsToTimes.pop();
                int repeat = times.pop();
                for(String element : tmp.keySet()) {
                    res.put(element, res.getOrDefault(element, 0) + tmp.get(element) * repeat);
                }
            } else {
                // c is a character
                String element = null;
                if(c >= 'a' && c <= 'z') {
                    i--;
                    element = formula.substring(i, i + 2);
                } else {
                    element = String.valueOf(c);
                }
                // we cannot ensure map do not contain this key so far, such as ON(ON)2
                res.put(element, res.getOrDefault(element, 0) + (k == 0 ? 1 : k));
                // after use k, we should reset k to 0
                k = 0;
                pow = 1;
            }
        }

        // construct our result
        StringBuilder sb = new StringBuilder();
        for(String key : res.keySet()) {
            sb.append(key);
            sb.append(res.get(key) == 1 ? "" : res.get(key));
        }
        return sb.toString();
    }

//    private boolean isBipartite(int[][] edges) {
//        if(edges == null || edges.length == 0) return true;
//        Map<Integer, Integer> nodeToColor = new HashMap<>();
//        Map<Integer, List<Integer>> adj = new HashMap<>();
//        HashSet<Integer> nodes = new HashSet<>();
//
//        for(int [] edge : edges) {
//            int node1 = edge[0], node2 = edge[1];
//            if(!adj.containsKey(node1)) {
//                adj.put(node1, new ArrayList<>());
//                nodes.add(node1);
//            }
//            if(!adj.containsKey(node2)) {
//                adj.put(node2, new ArrayList<>());
//                nodes.add(node2);
//            }
//            adj.get(node1).add(node2);
//            adj.get(node2).add(node1);
//        }
//
//        while(!nodes.isEmpty()) {
//            Iterator<Integer> iter = nodes.iterator();
//            int start = iter.next();
//            Queue<Integer> queue = new LinkedList<>();
//            queue.add(start);
//            nodes.remove(start);
//            nodeToColor.put(start, 1);
//
//            while(!queue.isEmpty()) {
//                int head = queue.poll();
//                int headColor = nodeToColor.get(head);
//                List<Integer> neighbours = adj.get(head);
//                for(int neigh : neighbours) {
//                    if(!nodeToColor.containsKey(neigh)) {
//                        nodeToColor.put(neigh, 1 - headColor);
//                        nodes.remove(neigh);
//                        queue.add(neigh);
//                    } else {
//                        if(nodeToColor.get(neigh) != 1 - headColor) {
//                            return false;
//                        }
//                    }
//                }
//            }
//        }
//        return true;
//    }



    public int amazingNumberFromWeb(int[] a) {
        int len = a.length;
        LinkedList<Interval> intervals = new LinkedList<>();

        // find all the intervals that if the array starts at any index in the interval, there will
        // be at least 1 element is amazing number
        for (int i = 0; i < len; i++) {
            if (a[i] >= len) continue;
            int start = (i+1) % len;
            int end = (len + (i - a[i])) % len;
            System.out.println(i + ": " + start + " - " + end);
            intervals.add(new Interval(start, end));
        }

        // now our problem has just become: "find the year that has the maximum number of people
        // alive"
        int[] count = new int[len];
        for (Interval i : intervals) {
            count[i.start]++;
            if (i.end+1 < count.length) count[i.end+1]--;
        }
        int max = 0;
        int counter = 0;
        int idx = 0;
        for (int i = 0; i < count.length; i++) {
            counter += count[i];
            if (counter > max) {
                max = counter;
                idx = i;
            }
        }

        return idx;
    }

    private int amazingNum(int[] nums) {
        int len = nums.length;
        int[] count = new int[len];
        for(int i = 0; i < len; i++) {
            int num = nums[i];
            if(num < 0 || num >= len) continue;
            int start = (i + 1) % len;
            int end = (len - num + i) % len;
            count[start]++;
            if(end + 1 < len) {
                count[end + 1]--;
            }
        }
        int max = 0, idx = 0, counter = 0;
        for(int i = 0; i < len; i++) {
            counter += count[i];
            if(counter > max) {
                idx = i;
                max = counter;
            }
        }
        return idx;
    }

    public int[] dailyTemperatures(int[] temperatures) {
        int len = temperatures.length;
        Stack<Integer> idxes = new Stack<>();
        int[] res = new int[len];

        for(int i = len - 1; i >= 0; i--) {
            if(idxes.isEmpty()) {
                idxes.push(i);
                res[i] = 0;
            } else {
                while(!idxes.isEmpty() && temperatures[i] >= temperatures[idxes.peek()]) {
                    idxes.pop();
                }
                if(idxes.isEmpty()) {
                    res[i] = 0;
                } else {
                    res[i] = idxes.peek() - i;
                }
                idxes.push(i);
            }
        }
        return res;
    }

    public int monotoneIncreasingDigits(int N) {
        if(N >= 0 && N <= 9) return N;
        // len >= 2
        char[] chars = String.valueOf(N).toCharArray();
        int len = chars.length;

        //fint idx that chars[i] > chars[i + 1]
        int idx = 0;
        for(; idx < len - 1; idx++) {
            if(chars[idx] > chars[idx + 1]) {
                break;
            }
        }
        if(idx == len - 1) {
            return N;
        } else {
            while(idx >= 1 && chars[idx] == chars[idx - 1]) {
                idx--;
            }
            chars[idx++] -= 1;
            while(idx < len) {
                chars[idx++] = '9';
            }
            return Integer.parseInt(new String(chars));
        }
    }

    public int deleteAndEarn(int[] nums) {
        Map<Integer, Integer> numToTimes = new HashMap<>();
        Map<Integer, Integer> takeCur = new HashMap<>();
        Map<Integer, Integer> notTakeCur = new HashMap<>();

        Set<Integer> uniqueNums = new HashSet<>();
        Arrays.sort(nums);
        for(int i = 0; i < nums.length; i++) {
            int num = nums[i];
            numToTimes.put(num, numToTimes.getOrDefault(num, 0) + 1);
            uniqueNums.add(num);
        }
        List<Integer> numsList = new ArrayList<>(uniqueNums);
        Collections.sort(numsList);

        takeCur.put(0, 0); // num - > benefit
        notTakeCur.put(0, 0);
        int prev = 0;
        for(int i = 0; i < numsList.size(); i++) {
            int cur = numsList.get(i);
            if(cur - prev == 1) {
                takeCur.put(cur, notTakeCur.get(prev) + cur * numToTimes.get(cur));
                notTakeCur.put(cur, Math.max(takeCur.get(prev), notTakeCur.get(prev)));
            } else {
                takeCur.put(cur, Math.max(takeCur.get(prev), notTakeCur.get(prev)) + cur * numToTimes.get(cur));
                notTakeCur.put(cur, Math.max(takeCur.get(prev), notTakeCur.get(prev)));
            }
            prev = cur;
        }
        return Math.max(takeCur.get(prev), notTakeCur.get(prev));
    }

    public int cherryPickup(int[][] grid) {
        int m = grid.length, n = grid[0].length;

        int count = 0;
        List<int[]>[][] paths = new List[m + 1][n + 1];

        // prev process, set all non-reachable cell to -1
        for(int i = 0; i < m; i++) {
            if(grid[i][0] == -1) {
                for(int j = i + 1; j < m; j++) {
                    grid[j][0] = -1;
                }
                break;
            }
        }

        for(int i = 0; i < n; i++) {
            if(grid[0][i] == - 1) {
                for(int j = i + 1; j < n; j++) {
                    grid[0][j] = -1;
                }
                break;
            }
        }

        for(int i = 0; i <= m; i++) paths[i][0] = new ArrayList<>();
        for(int j = 0; j <= n; j++) paths[0][j] = new ArrayList<>();

        for(int i = 1; i < m; i++) {
            for(int j = 1; j < n; j++) {
                if(grid[i - 1][j] == - 1 && grid[i][j - 1] == -1) {
                    grid[i][j] = -1;
                }
            }
        }

        // first pass
        for(int i = 1; i <= m; i++) {
            for(int j = 1; j <= n; j++) {
                if(grid[i - 1][j - 1] == -1) continue;

                if(paths[i - 1][j].size() > paths[i][j - 1].size()) {
                    paths[i][j] = new ArrayList<>(paths[i - 1][j]);
                } else {
                    paths[i][j] = new ArrayList<>(paths[i][j - 1]);
                }

                if(grid[i - 1][j - 1] == 1) {
                    paths[i][j].add(new int[]{i - 1, j - 1});
                }
            }
        }

        if(paths[m][n] == null) {
            return 0;
        } else {
            count =  paths[m][n].size();
            for(int[] coordinate : paths[m][n]) {
                grid[coordinate[0]][coordinate[1]] = 0;
            }
        }

        for(int i = 1; i <= m; i++) {
            for(int j = 1; j <= n; j++) {
                if(grid[i - 1][j - 1] == -1) continue;

                if(paths[i - 1][j].size() > paths[i][j - 1].size()) {
                    paths[i][j] = new ArrayList<>(paths[i - 1][j]);
                } else {
                    paths[i][j] = new ArrayList<>(paths[i][j - 1]);
                }

                if(grid[i - 1][j - 1] == 1) {
                    paths[i][j].add(new int[]{i - 1, j - 1});
                }
            }
        }
        int res = count + paths[m][n].size();
        return res;
    }

    public int[] maxSumOfThreeSubarrays3(int[] nums, int k) {
        int len = nums.length;
        int[] preSum = new int[len + 1];
        int sum = 0;
        for(int i = 0; i < len; i ++) {
            sum += nums[i];
            preSum[i + 1] = sum;
        }

        int[] left = new int[len];
        int max = 0;
        // for left, we use the end index
        for(int i = k - 1; i < len; i++) {
            int sub = preSum[i + 1] - preSum[i + 1 - k];
            if(sub > max) {
                max = sub;
                left[i] = i - k + 1;
            } else {
                left[i] = left[i - 1];
            }
        }

        // for right, we use start index
        max = 0;
        int[] right = new int[len];
        for(int i = len - k; i >= 0; i--) {
            int sub = preSum[i + k] - preSum[i];
            if(sub > max) {
                max = sub;
                right[i] = i;
            } else {
                right[i] = right[i + 1];
            }
        }

        max = 0;
        int[] res = new int[3];
        for(int i = k; i <= len - 2 * k; i++) {
            int leftIdx = left[i - 1];
            int rightIdx = right[i + k];
            int sub = preSum[i + k] - preSum[i] + preSum[leftIdx + k] - preSum[leftIdx] + preSum[rightIdx + k] - preSum[rightIdx];
            if(sub > max) {
                max = sub;
                res[0] = leftIdx;
                res[1] = i;
                res[2] = rightIdx;
            }
        }
        return res;
    }

    public String convert(String s, int numRows) {
        StringBuilder[] sbs = new StringBuilder[numRows];
        for(int i = 0; i < numRows; i++) {
            sbs[i] = new StringBuilder();
        }

        int index = 0;
        while(index < s.length()) {
            int rowIdx = 0;
            while(rowIdx < numRows && index < s.length()) {
                sbs[rowIdx++].append(s.charAt(index++));
            }
            rowIdx = numRows - 2;
            while(rowIdx > 0 && index < s.length()) {
                sbs[rowIdx--].append(s.charAt(index++));
            }
        }
        StringBuilder sb = new StringBuilder();
        for(StringBuilder tmp : sbs) {
            sb.append(tmp);
        }
        return sb.toString();
    }

    private List<String> permutation(String s) {
        List<String> res = new ArrayList<>();
        boolean[] visited = new boolean[4];
        permutationHelper(res, s, "", visited);
        return res;
    }

    private void permutationHelper(List<String> res, String s, String prev, boolean[] visited) {
        if(prev.length() == s.length()) {
            res.add(prev);
            return;
        }
        for(int i = 0; i < s.length(); i++) {
            if(!visited[i]) {
                visited[i] = true;
                String cur = prev + s.charAt(i);
                permutationHelper(res, s, cur, visited);
                visited[i] = false;
            }
        }
    }

    public String nextTime(String s) {
        s = s.substring(0, 2) + s.substring(3, 5);
        int curHour = Integer.parseInt(s.substring(0, 2));
        int curMin = Integer.parseInt(s.substring(2, 4));
        List<String> times = permutation(s);

        String res = null;
        int minDistance = Integer.MAX_VALUE;
        for(String time : times) {
            int hour = Integer.parseInt(time.substring(0, 2));
            int min = Integer.parseInt(time.substring(2, 4));
            if(hour >= 24 || min >= 60) {
                continue;
            } else {
                // hour <= 23 && min <= 59
                int distance = hour * 60 + min - (curHour * 60 + curMin);
                if(distance <= 0) {
                    distance += 60 * 24;
                }
                if(distance < minDistance) {
                    res = time.substring(0, 2) + ":" + time.substring(2, 4);
                    minDistance = distance;
                }
            }
        }
        return res;
    }


    public int flower(int[] P, int K) {
        // write your code in Java SE 8
        TreeSet<Integer> active = new TreeSet<>();

        for(int i = P.length - 1; i >=0; i--) {
            int num = P[i];
            active.add(num);
            int lower = 0, higher = P.length + 1;
            if(active.lower(num) != null) {
                lower = active.lower(num);
            }
            if(active.higher(num) != null) {
                higher = active.higher(num);
            }
            if(num - lower - 1 == K) {
                return i;
            }
            if(higher - num - 1 == K) {
                return i;
            }
        }
        return -1;
    }

    public int numJewelsInStones(String J, String S) {
        if(J == null || S == null) return 0;
        Set<Character> set = new HashSet<>();

        for(char c : J.toCharArray()) {
            set.add(c);
        }

        int res = 0;
        for(char c : S.toCharArray()) {
            if(set.contains(c)) {
                res++;
            }
        }
        return res;
    }

    public boolean isIdealPermutation(int[] A) {
        for(int i = 0; i < A.length - 1; i++) {
            if(A[i] > A[i + 1]) {
                int tmp = A[i]; //A[i] =
                A[i] = A[i + 1];
                A[i + 1] = tmp;
            }
        }

        for(int i = 0; i < A.length - 1; i++) {
            if(A[i] > A[i + 1]) {
                return false;
            }
        }
        return true;
    }

    public double minmaxGasDist(int[] stations, int K) {
        Arrays.sort(stations);

        double[] distance = new double[stations.length - 1];
        for(int i = 0; i < stations.length - 1; i++) {
            distance[i] = (double)(stations[i + 1] - stations[i]);
        }

        Arrays.sort(distance);
        double max = distance[distance.length - 1], min = distance[0];
        double thresh = max / (K + 1);

        double res = (double) Integer.MAX_VALUE;

        double from = thresh, to = max;
        while (to - from > Math.pow(10, -7)) {
            double tmp = (from + to) / 2;
            int seg = 0;
            for(int i = 0; i < distance.length; i++) {
                seg += Math.ceil(distance[i] / tmp);
            }
            if(seg > (K + distance.length)) {
                from = tmp;
            } else {
                to = tmp;
                res = tmp;
            }
        }
        return res;
    }

    static class Puzzle {
        
        int[][] puzzle;
        final int[][] dir = {{0,1}, {0, -1}, {1, 0}, {-1, 0}};

        public Puzzle(int[][] puzzle) {
            this.puzzle = puzzle;
        }

        public List<Puzzle> next() {
            // find slot
            int x = -1, y = -1;
            boolean find = false;
            for(int i = 0; i < this.puzzle.length && !find; i++) {
                for(int j = 0; j < this.puzzle[0].length && !find; j++) {
                    if(this.puzzle[i][j] == 0) {
                        x = i;
                        y = j;
                        find = true;
                    }
                }
            }

            List<Puzzle> res = new ArrayList<>();
            for(int i = 0; i < 4; i++) {
                int xx = dir[i][0] + x, yy = dir[i][1] + y;
                if(xx >= 0 && xx < 2 && yy >= 0 && yy < 3) {
                    int[][] nextPuzzle = new int[2][3];
                    for(int k = 0; k < 2; k++) {
                        for(int l = 0; l < 3; l++) {
                            nextPuzzle[k][l] = this.puzzle[k][l];
                        }
                    }
                    nextPuzzle[x][y] = this.puzzle[xx][yy];
                    nextPuzzle[xx][yy] = 0;
                    res.add(new Puzzle(nextPuzzle));
                }
            }
            return res;
        }
        @Override
        public boolean equals(Object p) {
            for(int i = 0; i < 2; i++) {
                for(int j = 0; j < 3; j++) {
                    if(this.puzzle[i][j] != ((Puzzle)p).puzzle[i][j]) {
                        return false;
                    }
                }
            }
            return true;
        }

        @Override
        public int hashCode() {
            return 1;
        }
    }

    public int slidingPuzzle(int[][] board) {
        int[][] des = {{1,2,3}, {4,5,0}};
        Puzzle begin = new Puzzle(board);
        Puzzle end = new Puzzle(des);

        if(begin.equals(end)) {
            return 0;
        } else {

            Queue<Puzzle> from = new LinkedList<>();
            Queue<Puzzle> target = new LinkedList<>();
            Set<Puzzle> visited = new HashSet<>();

            from.add(begin);
            target.add(end);
            int step = 1;

            while(!from.isEmpty() && !target.isEmpty()) {
                Queue<Puzzle> q = null;
                Queue<Puzzle> residual = null;

                if(from.size() < target.size()) {
                    q = from;
                    residual = target;
                } else {
                    q = target;
                    residual = from;
                }

                int size = q.size();

                while(size-- > 0) {
                    Puzzle top = q.poll();
                    visited.add(top);

                    List<Puzzle> nextSet = top.next();
                    for(Puzzle puzzle : nextSet) {
                        if(residual.contains(puzzle)) {
                            return step;
                        } else {
                            if(!visited.contains(puzzle)) {
                                q.add(puzzle);
                            }
                        }
                    }
                }
                step++;
            }
            return -1;
        }
    }

    public List<Integer> findSubstring(String s, String[] words) {
        Map<String, Integer> map = new HashMap<>();
        List<Integer> res = new ArrayList<>();

        int remain = words.length, len = words[0].length();
        for(String word : words) {
            map.put(word, 1);
        }

        int fast = 0, slow = 0;
        while(fast <= s.length() - len) {
            String word = s.substring(fast, fast + len);
            if(map.keySet().contains(word)) {
                int times = map.get(word);
                if(times == 1) {
                    remain--;
                }
                map.put(word, map.get(word) - 1);
            }
            
            while(remain == 0) {
                if(fast - slow == len * (words.length - 1)) {
                    res.add(slow);
                    String w = s.substring(slow, slow + len);
                    map.put(w, map.get(w) + 1);
                    remain += 1;
                    
                } else if(fast - slow > len * (words.length - 1)) {
                    String w = s.substring(slow, slow + len);
                    if(map.keySet().contains(w)) {
                        map.put(w, map.get(w) + 1);
                        if(map.get(w) == 1) {
                            remain += 1;
                        }
                    }
                } else {
                    String w = s.substring(slow, slow + len);
                    if(map.keySet().contains(w)) {
                        map.put(w, map.get(w) + 1);
                        if(map.get(w) == 1) {
                            remain += 1;
                        }
                    }
                }
                slow++;
            }
            fast++;
        }
        return res;
    }

    public int kthGrammar(int N, int K) {
        if(N == 1 && K == 1) {
            return 1;
        }

        if(K > Math.pow(2, N - 2)) {
            return kthGrammarHelper(N - 1, (int)(K - Math.pow(2, N - 2)), true);
        } else {
            return kthGrammarHelper(N - 1, (int)(K), false);
        }
    }

    private int kthGrammarHelper(int N, int K, boolean invert) {
        if(N == 1) {
            if(invert) {
                return 1;
            } else {
                return 0;
            }
        }

        if(K > Math.pow(2, N - 2)) {
            return kthGrammarHelper(N - 1, (int)(K - Math.pow(2, N - 2)), true ^ invert);
        } else {
            return kthGrammarHelper(N - 1, (int)(K), false ^ invert);
        }
    }

    public int repeatedStringMatch(String A, String B) {
        int i = 0, count = 0;
        String start = "";
        while (i < B.length()) {
            int idx = B.indexOf(A, i);
            if (idx == -1) break;
            if(i == 0) {
                start = B.substring(0, idx);
            }
            i = idx + A.length();
            count++;
        }
        B = B.replaceAll(A, ""); // remaining B if valid, should be smaller than A

        if(start.length() > 0) {
            StringBuilder sbA = new StringBuilder(A);
            String reverseA = sbA.reverse().toString();
            if(reverseA.startsWith(new StringBuilder(start).reverse().toString())) {
                count++;
                B = B.replace(start, "");
            } else {
                return -1;
            }
        }
        if(B.length() > 0) {
            if(A.startsWith(B)) {
                count++;
            } else {
                return -1;
            }
        }
        return count;
    }

    public int wordsTyping(String[] sentence, int rows, int cols) {
        String s = String.join(" ", sentence) + " ";
        int start = 0, l = s.length();
        for (int i = 0; i < rows; i++) {
            start += cols;
            if (s.charAt(start % l) == ' ') {
                start++;
            } else {
                while (start > 0 && s.charAt((start-1) % l) != ' ') {
                    start--;
                }
            }
        }

        return start / s.length();
    }

    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String, String> parents = new HashMap<>();
        Map<String, String> owners = new HashMap<>();

        for(List<String> account : accounts) {
            String owner = account.get(0);
            for(int i = 1; i < account.size(); i++) {
                if(i == 1) {
                    parents.put(account.get(i),  account.get(i));
                } else {
                    parents.put(account.get(i), account.get(1));
                }
                owners.put(account.get(i), owner);
            }
        }

        for(List<String> account : accounts) {
            for(int i = 1; i < account.size(); i++) {
                parents.put(account.get(i), find(account.get(i), parents));
            }
        }

        Map<String, TreeSet<String>> union = new HashMap<>();

        for(List<String> account : accounts) {
            for(int i = 1; i < account.size(); i++) {
                String parent = find(account.get(i), parents);
                if(!union.keySet().contains(parent)) {
                    union.put(parent, new TreeSet<>());
                }
                union.get(parent).add(account.get(i));
            }
        }


        List<List<String>> res = new ArrayList<>();
        for(String addr : union.keySet()) {
            List<String> list = new ArrayList<>(union.get(addr));
            list.add(0, owners.get(addr));
            res.add(list);
        }
        return res;
    }

    private String find(String addr, Map<String, String> parents) {
        return parents.get(addr).equals(addr) ?  addr : find(parents.get(addr), parents);
    }

    public String encode(String s) {
        String[][] dp = new String[s.length()][s.length()];

        for(int l = 1; l <= s.length(); l++) {
            for(int i = 0; i <= s.length() - l; i++) {

                int j = i + l - 1;
                if(l < 5) {
                    dp[i][j] = s.substring(i, j + 1);
                } else {
                    String substr = s.substring(i, j + 1);

                    dp[i][j] = substr;
                    for(int k = i; k < j; k++) {
                        if(dp[i][k].length() + dp[k + 1][j].length() < dp[i][j].length()) {
                            dp[i][j] = dp[i][k] + dp[k + 1][j];
                        }
                    }

                    for(int sl = 1; sl <= l / 2; sl++) {
                        String ss = substr.substring(0, sl);
                        if(l % sl == 0 && substr.replaceAll(ss, "").length() == 0) {
                            String tmp = (l / sl) + "[" + dp[i][i + sl - 1] + "]";
                            if(tmp.length() < dp[i][j].length()) {
                                dp[i][j] = tmp;
                            }
                        }
                    }
                }
            }
        }
        return dp[0][s.length() - 1];
    }

    public TreeNode[] splitBST(TreeNode root, int V) {
        TreeNode[] res = new TreeNode[2];

        if(root == null) {
            return new TreeNode[]{null, null};
        }

        if(root.val <= V) {
            res[0] = root;
            TreeNode[] subres = splitBST(root.right, V);
            root.right = subres[0];
            res[1] = subres[1];
        } else {
            // root.val > v
            res[1] = root;
            TreeNode[] subres = splitBST(root.left, V);
            root.left = subres[1];
            res[0] = subres[0];
        }
        return res;
    }

    public int minDiffInBST(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        // to left most node
        while(p != null) {
            stack.push(p);
            p = p.left;
        }

        int prev = -1, minDistance = Integer.MAX_VALUE;
        boolean assign = false;

        while(!stack.isEmpty()) {
            p = stack.pop();
            if(!assign) {
                prev = p.val;
                assign = true;
            } else {
                int cur = p.val;
                minDistance = Math.min(minDistance, cur - prev);
                prev = cur;
            }
            //System.out.print(p.val + "  ");
            if (p.right != null) {
                p = p.right;
                while (p != null) {
                    stack.push(p);
                    p = p.left;
                }
            }
        }
        return minDistance;
    }

    public int numRabbits(int[] answers) {
        if(answers == null || answers.length == 0) return 0;

        Map<Integer, Integer> map = new HashMap<>();

        for(int num : answers) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        int sum = 0;
        for(int key : map.keySet()) {
            int others = map.get(key);

            sum += ((int)Math.ceil((double)others / (double)(key + 1)))* (key + 1);
        }
        return sum;
    }

    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
        if(sx == tx) {
            if(ty % sy == 0) {
                return true;
            }
        } else if(sy == ty) {
            if(tx % sx == 0) {
                return true;
            }
        }

        if(sx == tx && sy == ty) {
            return true;
        } else if(sx > tx || sy > ty) {
            return false;
        }
        if(tx > ty) {
            return reachingPoints(sx, sy, tx - ty, ty);
        } else {
            return reachingPoints(sx, sy, tx, ty - tx);
        }
    }

    class Board {

        int[][] board;

        public Board(int[][] board) {
            this.board = board;
        }

        boolean isChessBoard() {
            int start = board[0][0];
            for(int i = 0; i < board.length; i++) {
                for(int j = 0; j < board[0].length; j++) {
                    if((((i + j) % 2) == 0) && board[i][j] != board[0][0]) {
                        return false;
                    } else if ((((i + j) % 1) == 0) && board[i][j] == board[0][0]) {
                        return false;
                    }
                }
            }
            return true;
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            for(int i = 0; i < board.length; i++) {
                for(int j = 0; j < board[0].length; j++) {
                    sb.append(board[i][j]);
                }
            }
            return board.toString();
        }
    }

    final char[] oper = new char[]{'+', '-', '*', '/'};

    private List<List<Character>>getOper() {
        List<Character> list = new ArrayList<>();
        List<List<Character>> res = new ArrayList<>();
        getOperHelper(res, list, 3);
        return res;
    }

    private void getOperHelper(List<List<Character>> res, List<Character> prev, int remain) {
        if(remain == 0) {
            res.add(prev);
            return;
        }

        for(int i = 0; i < 4; i++) {
            List<Character> cur = new ArrayList<>(prev);
            cur.add(oper[i]);
            getOperHelper(res, cur, remain - 1);
        }
    }

    private List<List<Integer>> getPermutation(int[] nums) {
        int len = nums.length;
        boolean[] visited = new boolean[len];
        List<Integer> list = new ArrayList<>();
        List<List<Integer>> res = new ArrayList<>();
        permutationHelper(res, list, visited, nums.length, nums);
        return res;
    }

    private void permutationHelper(List<List<Integer>> res, List<Integer> prev, boolean[] visited, int remain, int[] nums) {
        if(remain == 0) {
            res.add(new ArrayList<>(prev));
            return;
        }

        for(int i = 0; i < nums.length; i++) {
            if(!visited[i]) {
                prev.add(nums[i]);
                visited[i] = true;
                permutationHelper(res, prev, visited, remain - 1, nums);
                visited[i] = false;
                prev.remove(prev.size() - 1);
            }
        }
    }

    public boolean judgePoint24(int[] nums) {
        List<List<Character>> opers = getOper();
        List<List<Integer>> operands = getPermutation(nums);

        int[] operOrder = new int[]{0, 1, 2};
        List<List<Integer>> operOder = getPermutation(operOrder);

        for(int i = 0; i < operOder.size(); i++) {
            for(int j = 0; j < operands.size(); j++) {
                for(int k = 0; k < opers.size(); k++) {
                    double tmp = calculate(opers.get(k), operOder.get(i), operands.get(j));
                    if(Math.abs(tmp - 24) < 0.01) {
                        System.out.println(opers.get(k));
                        System.out.println(operOder.get(i));
                        System.out.println(operands.get(j));
                        System.out.println(tmp);
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private double calculate(List<Character> oper, List<Integer> operOrder, List<Integer> operand) {
        double res = 0;
        for(int i = 0; i < operOrder.size(); i++) {
            char c = oper.get(operOrder.get(i));
            if(c == '+') {
                res = (double)operand.get(i) + (double)operand.get(i + 1) + res;
            } else if(c == '-') {
                res = (double)operand.get(i) - (double)operand.get(i + 1) + res;
            } else if(c == '*') {
                res = (double)operand.get(i) * (double)operand.get(i + 1) + res;
            } else {
                res = (double)operand.get(i) / (double)operand.get(i + 1) + res;
            }
        }
        return res;
    }

    public boolean canTransform(String start, String end) {
        if(start.length() != end.length()) return false;

        int ps = 0, pe = 0;
        while(ps < start.length()) {
            while (ps < start.length() && (start.charAt(ps) != 'R' && start.charAt(ps) != 'L')) ps++;
            while (pe < end.length() && (end.charAt(pe) != 'R' && end.charAt(pe) != 'L')) pe++;

            char cs = ps < start.length() ? start.charAt(ps) : 'x';
            char ce = pe < end.length() ? end.charAt(pe) : 'x';
            if(cs != ce) {
                return false;
            } else {
                if(ps > pe) {
                    if(cs == 'L') {
                        String sub = end.substring(pe + 1, ps + 1);
                        if(sub.replaceAll("X", "").length() != 0) {
                            return false;
                        } else {
                            pe = ps;
                        }
                    } else {
                        return false;
                    }
                } else if(ps < pe) {
                    if(cs == 'R') {
                        String sub = start.substring(ps + 1, pe + 1);
                        if(sub.replaceAll("X", "").length() != 0) {
                            return false;
                        } else {
                            ps = pe;
                        }
                    } else {
                        return false;
                    }
                }
            }
            ps++;
            pe++;
        }
        return true;
    }

    public List<String> letterCasePermutation(String S) {
        List<String> res = new ArrayList<>();
        if(S == null || S.length() == 0) {
            if(S != null) {
                res.add("");
            }
            return res;
        }

        letterCasePermutationHelper(S, res, 0, S);
        return res;
    }

    private void letterCasePermutationHelper(String s, List<String> res, int start, String prev) {
        if(start == s.length()) {
            res.add(prev);
            return;
        }
        for(int i = start; i < s.length(); i++) {
            char c = s.charAt(i);
            if(c >= 'a' && c <= 'z') {
                letterCasePermutationHelper(s, res, i + 1, prev);
                letterCasePermutationHelper(s, res, i + 1, prev.substring(0, i) +
                        (char)('A' + c - 'a') + prev.substring(i + 1));
                break;
            }
            if(c >= 'A' && c <= 'Z') {
                letterCasePermutationHelper(s, res, i + 1, prev);
                letterCasePermutationHelper(s, res, i + 1, prev.substring(0, i) +
                        (char)('a' + c - 'A') + prev.substring(i + 1));
                break;
            }
            if(i == s.length() - 1) {
                res.add(prev);
            }
        }
    }

    public boolean isBipartite(int[][] graph) {
        int len = graph.length;
        int[] color = new int[len];
        Arrays.fill(color, -1);

        for(int i = 0; i < len; i++) {
            int src = i, srcColor = 0;
            if(color[src] != -1) {
                srcColor = color[src];
            }

            for(int j = 0; graph[i] != null && j < graph[i].length; j++) {
                if(color[graph[i][j]] == srcColor) {
                    return false;
                } else if(color[graph[i][j]] == -1) {
                    color[graph[i][j]] = 1 - srcColor;
                }
            }
        }
        return true;
    }

    class Pair {
        int price;
        int dest;
        public Pair(int dest, int price) {
            this.dest = dest;
            this.price = price;
        }
    }

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        int[][] prices = new int[n][n];
        for(int [] flight : flights) {
            int from = flight[0], to = flight[1], price = flight[2];
            prices[from][to] = price;
        }

        int[] res = new int[n];
        Arrays.fill(res, Integer.MAX_VALUE);
        res[src] = 0;

        int step = K + 1;
        Queue<Integer> froms = new LinkedList<>();
        froms.add(src);

        while(step-- > 0) {
            int size = froms.size();
            while (size-- > 0) {
                int from = froms.poll();
                for(int i = 0; i < n; i++) {
                    if(prices[from][i] != 0 && from != i) {
                        int newPrice = prices[from][i] + res[from];
                        if(res[i] > newPrice) {
                            res[i] = newPrice;
                            froms.add(i);
                        }
                    }
                }
            }
        }

        return res[dst] == Integer.MAX_VALUE ? -1 : res[dst];
    }

    public int[] kthSmallestPrimeFraction(int[] A, int K) {
        int len = A.length;
        int aa = 5;
        return null;
    }

    public static void main(String[] args) {

        Solution solution = new Solution();
        TreeNode node1 = new TreeNode(1);
        TreeNode node2 = new TreeNode(2);
        TreeNode node3 = new TreeNode(3);
        TreeNode node4 = new TreeNode(4);
        TreeNode node5 = new TreeNode(5);
        TreeNode node6 = new TreeNode(6);
        TreeNode node7 = new TreeNode(7);
        TreeNode node8 = new TreeNode(8);
        TreeNode node9 = new TreeNode(9);
        TreeNode node10 = new TreeNode(10);
        TreeNode node11 = new TreeNode(11);


        node1.left = node2;
        node1.right = node3;
        node2.left = node4;
        node2.right = node5;
        node3.right = node6;
        node4.left = node7;
        node4.right = node8;
        node6.left = node9;
        node5.left = node11;
        node6.right = node10;


        Map<Integer, Integer> map = new HashMap<>();
        map.remove(5);

        int[][] image = new int[][]{{1,1,1}, {1,1,0}, {1,0,1}};
        String[][] pairs = new String[][]{{"great", "fine"}, {"acting","drama"}, {"skills","talent"}, {"great", "exce"}};
        String[] word1 = new String[]{"great", "acting", "skills"};
        String[] word2 = new String[]{"fine", "drama", "talent"};
        int[] asteroids = new int[]{-2,1,-1,-2};
        int[][] edges = new int[][]{{1,5}, {4,5}, {2,3}, {1, 4}};
        int[][] grid = {{1,1,1,1,0,0,0},
                {0,0,0,1,0,0,0},{0,0,0,1,0,0,1},{1,0,0,1,0,0,0},
                {0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,1,1,1}};
        int [] a = new int[]{1,2,1,2,6,7,5,1};
        //solution.postOrderIterOneStack(node1);//Stack(node1);//orderIter(node1);//rderIter(node1);
        int[] flower = {3,1,5,4,2};
        int[][] fff = {{1,2,3}, {5,4,0}};
        int[][] ff = {{1,2,3}, {5,4,0}};
        Puzzle p = new Puzzle(fff);
        Puzzle pp = new Puzzle(ff);
        Set<Puzzle> set = new HashSet<>();
        set.add(p);
        String[] words = {"hello", "world"};

        List<String> l1 = Arrays.asList("Gabe","Gabe2@m.co","Gabe3@m.co","Gabe4@m.co");
        List<String> l2 = Arrays.asList("o1", "e4", "e5", "e6");

        List<List<String>> ll = new ArrayList<>();
        ll.add(l1); ll.add(l2);

        int[] rabbits = new int[]{1,7,23,29,47};
        List<Character> list1 = Arrays.asList('/', '+', '+');
        List<Integer> order = Arrays.asList(0, 1, 2);
        List<Integer> nums = Arrays.asList(1,5,9,1);

        System.out.println(solution.kthSmallestPrimeFraction(rabbits, 8));
    }
}
