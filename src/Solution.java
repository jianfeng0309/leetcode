
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

//    static int maximumDifference(int nodes, int[] a, int[] b) {
//        int result = Integer.MIN_VALUE;
//        int[] father = new int[nodes + 1];
//        for (int i = 0; i < father.length; i++) {
//            father[i] = i;
//        }
//        for (int i = 0; i < a.length; i++) {
//            union(a[i], b[i], father);
//        }
//        for (int i = 1; i < father.length; i++) {
//            result = Math.max(result, i - find(i, father));
//        }
//        return result;
//    }

//    public static int find(int x, int[] father) {
//        if (father[x] == x) {
//            return x;
//        }
//        father[x] = find(father[x], father);
//        return father[x];
//    }

//    public static void union(int a, int b, int[] father) {
//        int root_a = find(a, father);
//        int root_b = find(b, father);
//        if (root_a != root_b) {
//            if (root_a > root_b) {
//                father[root_a] = root_b;
//            } else {
//                father[root_b] = root_a;
//            }
//        }
//    }

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

//    public ListNode reverseKGroup(ListNode head, int k) {
//        if(k == 1 || head ==null) {
//            return head;
//        }
//
//        ListNode dommy = new ListNode(-1), fast = dommy, slow = dommy;
//        dommy.next =  head;
//        int idx = 0;
//        while(idx == 0) {
//            while(idx++ < k) {
//                // move to the group end
//                if(fast.next != null) {
//                    fast = fast.next;
//                } else {
//                    // fast.next == null, reach the end
//                    return dommy.next;
//                }
//            }
//            // reverse the group
//            ListNode prev = slow;
//            slow = slow.next;
//            slow = reverse(prev, slow, fast);
//            fast = slow;
//            idx = 0;
//        }
//
//        return dommy.next;
//    }

//    private ListNode reverse(ListNode prev, ListNode start, ListNode end) {
//        ListNode startnext = start.next;
//        while(start != end) {
//            ListNode nextnext = startnext.next;
//            startnext.next = start;
//            start = startnext;
//            startnext = nextnext;
//        }
//        prev.next.next = startnext;
//        ListNode res = prev.next;
//        prev.next = end;
//        return res;
//    }

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

    public int countNodes(TreeNode root) {
        int h = height(root);
        return h < 0 ? 0 : height(root.right) == h - 1 ? (1 << h) + countNodes(root.right) : (1 << (h - 1)) + countNodes(root.left);
    }

    private int height(TreeNode root) {
        return root == null ? -1 : 1 + height(root.left);
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
        Queue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return A[o1[0]] * A[o2[1]] - A[o1[1]] * A[o2[0]];
            }
        });

        int[] Ksmallest;
        for(int i = 0; i < A.length - 1; i++) {
            pq.add(new int[]{i, A.length - 1});
        }

        while(K-- > 0) {
            Ksmallest = pq.poll();
            if(Ksmallest[1] - Ksmallest[0] > 1) {
                pq.add(new int[]{Ksmallest[0], Ksmallest[1] - 1});
            }
        }

        int[] pos = pq.peek();
        return new int[]{A[pos[0]], A[pos[1]]};
    }

    class Node {
        // the value of this node
        int val;
        // duplication of this node, 1 for single copy, not 1 dup
        int dup;
        // how many nodes that are smaller than the value of this node
        int count;

        public Node (int val) {
            this.val = val;
            dup = 1;
            count = 0;
        }
    }

    public int rotatedDigits(int N) {
        int count = 0;
        for(int i = 1; i <= N; i++) {
            if(isValid(i)) {
                count++;
            }
        }
        return count;
    }

    private boolean isValid(int num) {
        String original = String.valueOf(num);
        StringBuilder sb = new StringBuilder();
        for(char c : original.toCharArray()) {
            if (c == '0' || c == '1' || c== '8') {
                sb.append(c);
            } else if(c == '2') {
                sb.append('5');
            } else if(c == '5') {
                sb.append('2');
            } else if(c == '6') {
                sb.append('9');
            } else if(c == '9') {
                sb.append('6');
            } else {
                return false;
            }
        }
        return !sb.toString().equals(original);
    }

    public boolean escapeGhosts(int[][] ghosts, int[] target) {
        int OurDistance = Math.abs(target[0]) + Math.abs(target[1]);
        for(int [] ghost : ghosts) {
            int ghostDistance = Math.abs(target[0] - ghost[0]) + Math.abs(target[1] - ghost[1]);
            if(ghostDistance <= OurDistance) {
                return false;
            }
        }
        return true;
    }

    public String customSortString(String s, String t) {
        if(s == null || s.length() == 0 || t == null || t.length() == 0) return t;

        Map<Character, Integer> charToNum = new HashMap<>();
        Map<Integer, Character> numToChar = new HashMap<>();
        for(int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            charToNum.put(c, i);
            numToChar.put(i, c);
        }

        List<Integer> list = new ArrayList<>();
        StringBuilder res = new StringBuilder();

        for(int i = 0; i < t.length(); i++) {
            char tc = t.charAt(i);
            if(charToNum.containsKey(tc)) {
                list.add(charToNum.get(tc));
            } else {
                res.append(tc);
            }
        }

        Collections.sort(list);

        for(int num : list) {
            res.append(numToChar.get(num));
        }
        return res.toString();

    }

    public int numTilings(int N) {
        if(N == 1) {
            return 1;
        } else if(N == 2) {
            return 2;
        }
        long[] fulldp = new long[N];
        long[] omitdp = new long[N];
        fulldp[0] = 1;
        fulldp[1] = 2;
        omitdp[1] = 2;

        for(int i = 2; i < N; i++) {
            fulldp[i] = (fulldp[i - 2] + fulldp[i - 1]) % M + omitdp[i - 1] % M;
            omitdp[i] = omitdp[i - 1] % M + (fulldp[i - 2] * 2) % M;
        }
        return (int)(fulldp[N - 1] % M);
    }

    public boolean rotateString(String A, String B) {
        if(A == null && B == null){
            return true;
        } else if(A == null || B == null) {
            return false;
        }
        String AA = A + A;  
        int len = A.length();
        for(int i = 0; i < len; i++) {
            if(B.equals(AA.substring(i, i + len))) {
                return true;
            }
        }
        return false;
    }

    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        if(graph == null || graph.length == 0 || graph[0].length == 0) {
            return res;
        }
        int last = graph.length - 1;
        List<Integer> path = new ArrayList<>();
        path.add(0);
        allPathHelper(res, 0, path, last, graph);
        return res;
    }

    private void allPathHelper(List<List<Integer>> res, int prev, List<Integer> path, int last, int[][] graph) {
        if(prev == last) {
            res.add(new ArrayList<>(path));
            return;
        }

        for(int i = 0; i < graph[prev].length; i++) {
            int nextNode = graph[prev][i];
            path.add(nextNode);
            allPathHelper(res, nextNode, path, last, graph);
            path.remove(path.size() - 1);
        }
    }

    public double champagneTower(int poured, int query_row, int query_glass) {
        List<List<Double>> in = new ArrayList<>();
        List<List<Double>> out = new ArrayList<>();

        List<Double> inZero = new ArrayList<>();
        inZero.add((double)poured);
        List<Double> outZero = new ArrayList<>();
        outZero.add(Math.max(0.0, (double)poured - 1));
        in.add(inZero);
        out.add(outZero);

        if(query_row == 0 && query_glass == 0) {
            return Math.min(1.0, (double)poured);
        }

        for(int i = 1; i <= query_row; i++) {
            champangeTowerHelper(i, in, out);
        }
        return Math.min(1, in.get(query_row).get(query_glass));
    }

    private void champangeTowerHelper(int row, List<List<Double>> in, List<List<Double>> out) {
        List<Double> crtIn = new ArrayList<>();
        List<Double> crtOut = new ArrayList<>();
        for(int i = 0; i <= row; i++) {
            double left = i - 1 >= 0 ? out.get(row - 1).get(i - 1)/ 2 : 0;
            double right = i < out.get(row - 1).size() ? out.get(row - 1).get(i) / 2 : 0;
            crtIn.add(left + right);
            crtOut.add(Math.max(left + right - 1, 0.0));
        }
        in.add(crtIn);
        out.add(crtOut);
    }

    public String similarRGB(String color) {
        StringBuilder sb = new StringBuilder();
        sb.append("#");
        for(int i = 1; i < 6; i = i + 2) {
            String singleColor = color.substring(i, i + 2);
            sb.append(RGBhelper(singleColor));
        }
        return sb.toString();
    }

    private char RGBhelper(String singleColor) {
        if(singleColor.charAt(0) == singleColor.charAt(1)) {
            return singleColor.charAt(0);
        }

        int color = 0;
        for(char c : singleColor.toCharArray()) {
            if(c >= 'a' && c <= 'f') {
                color = color * 16 + (c - 'a') + 10;
            } else if(c >= '0' && c <= '9') {
                color = color * 16 + (c - '0');
            }
        }

        int minDis = Integer.MAX_VALUE;
        int index = -1;
        for(int i = 0; i < 16; i++) {
            int choice = i * 16 + i;
            int dis = Math.abs(choice - color);
            if(dis < minDis) {
                minDis = dis;
                index = i;
            }
        }
        if(index >= 0 && index <= 9) {
            return (char)('0' + index);
        } else {
            return (char)('a' + index - 10);
        }
    }

    public int minSwap(int[] A, int[] B) {
        int left = 0, right = 0;
        int len = A.length;
        if(len == 1) {
            return 0;
        }
        // len >= 2

        int[] Acp = new int[A.length];
        int[] Bcp = new int[B.length];
        for(int i = 0; i < len; i++) {
            Acp[i] = A[i];
            Bcp[i] = B[i];
        }

        for(int i = 1; i < len; i++) {
            if(A[i] > A[i - 1] && B[i] > B[i - 1]) {
                continue;
            } else {
                System.out.println( "left swap " + i);
                int tmp = A[i];
                A[i] = B[i];
                B[i] = tmp;
                left++;
            }
        }

        for(int i = len - 1; i >= 1; i--) {
            if(Acp[i] > Acp[i - 1] && Bcp[i] > Bcp[i - 1]) {
                continue;
            } else {
                System.out.println( "right swap " + i);
                int tmp = Acp[i];
                Acp[i] = Bcp[i];
                Bcp[i] = tmp;
                right++;
            }
        }
        return Math.min(left, right);
    }

    public List<Integer> eventualSafeNodes(int[][] graph) {
        int len = graph.length;

        List<List<Integer>> adj = new ArrayList<>();
        int[] in = new int[len];
        for(int i = 0; i < len; i++) {
            adj.add(new ArrayList<>());
        }
        for(int i = 0; i < len; i++) {
            int[] neighbours = graph[i];
            for(int neigh : neighbours) {
                adj.get(neigh).add(i);
                in[i]++;
            }
        }
        List<Integer> res = new ArrayList<>();
        Queue<Integer> q = new LinkedList<>();
        for(int i = 0; i < len; i++) {
            if(in[i] == 0) {
                q.add(i);
            }
        }
        while(!q.isEmpty()) {
            int node = q.poll();
            res.add(node);
            List<Integer> neighbours = adj.get(node);
            for(int neigh : neighbours) {
                in[neigh]--;
                if(in[neigh] == 0) {
                    q.add(neigh);
                }
            }
        }
        Collections.sort(res);
        return res;
    }

    private int BFS(int[][] grid, int[] pos) {
        Queue<int[]> q = new LinkedList<>();
        q.add(pos);
        grid[pos[0]][pos[1]] = -1;

        int res = 0;
        while(!q.isEmpty()) {
            int[] head = q.poll();
            //trace.add(head);
            int x = head[0], y = head[1];
            if(x == 0) {
                return 0;
            }
            //grid[x][y] = -1;
            res++;
            for(int i = 0; i < 4; i++) {
                int nextX = x + dirs[i][0];
                int nextY = y + dirs[i][1];
                if(nextX >= 0 && nextX < grid.length && nextY >= 0 && nextY < grid[0].length && grid[nextX][nextY] == 1) {
                    q.add(new int[]{nextX, nextY});
                    grid[nextX][nextY] = -1;
                }
            }
        }
        return res;
    }

    private void cleanUp(int[][] grid, boolean clean)  {
        for(int i = 0; i < grid.length; i++) {
            for(int j = 0; j < grid[0].length; j++) {
                if(grid[i][j] == -1) {
                    grid[i][j] = clean ? 0 : 1;
                }
            }
        }
    }

    public int[] maxSlidingWindow(int[] a, int k) {
        if (a == null || k <= 0) {
            return new int[0];
        }
        int n = a.length;
        int[] r = new int[n-k+1];
        int ri = 0;
        // store index
        Deque<Integer> q = new ArrayDeque<>();
        for (int i = 0; i < a.length; i++) {
            // remove numbers out of range k
            while (!q.isEmpty() && q.peek() < i - k + 1) {
                q.poll();
            }
            // remove smaller numbers in k range as they are useless
            while (!q.isEmpty() && a[q.peekLast()] < a[i]) {
                q.pollLast();
            }
            // q contains index... r contains content
            q.offer(i);
            if (i >= k - 1) {
                r[ri++] = a[q.peek()];
            }
        }
        return r;
    }

    public int totalTimeRunning(Map<Character, Integer> cooldown, String tasks) {
        Map<Character, Integer> startTime = new HashMap<>();
        int time = 0;

        for(int i = 0; i < tasks.length(); i++) {
            char task = tasks.charAt(i);
            if(startTime.containsKey(task)) {
                time = Math.max(time, startTime.get(task));
            }
            startTime.put(task, time + cooldown.getOrDefault(task, 0) + 1);
            if(i != tasks.length() - 1) {
                time = time + 1;
            }
        }
        return time;
    }

    public int minSwap801(int[] A, int[] B) {
        int len = A.length;
        int[] swap = new int[len];
        int[] non_swap = new int[len];
        Arrays.fill(swap, Integer.MAX_VALUE);
        Arrays.fill(non_swap, Integer.MAX_VALUE);

        swap[0] = 1;
        non_swap[0] = 0;

        for(int i = 1; i < len; i++) {
            if(A[i] > A[i - 1] && B[i] > B[i - 1]) {
                swap[i] = swap[i - 1] + 1;
                non_swap[i] = non_swap[i - 1];
            }

            if(A[i] > B[i - 1] && B[i] > A[i - 1]) {
                swap[i] = Math.min(non_swap[i - 1] + 1, swap[i]);
                non_swap[i] = Math.min(swap[i - 1], non_swap[i]);
            }
        }
        return Math.min(swap[len - 1], non_swap[len - 1]);
    }

    private long numOfZero(long num) {
        if(num == 0) {
            return 0;
        }

        return num / 5 + numOfZero(num / 5);
    }

    public int uniqueMorseRepresentations(String[] words) {
        if(words == null || words.length == 0) return 0;
        String[] morse = {".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."};
        Set<String> encodedWords = new HashSet<>();
        for(String word : words) {
            StringBuilder sb = new StringBuilder();
            for (char c : word.toCharArray()) {
                sb.append(morse[c - 'a']);
            }
            encodedWords.add(sb.toString());
        }
        return encodedWords.size();
    }

    public int[] numberOfLines(int[] widths, String S) {
        int lines = 1, pos = 0;
        for(char c : S.toCharArray()) {
            int space = widths[c - 'a'];
            if(pos + space <= 100) {
                pos += space;
            } else {
                pos = space;
                lines++;
            }
        }
        return new int[]{lines, pos};
    }

    public int maxIncreaseKeepingSkyline(int[][] grid) {
        int len = grid.length;
        int[] rowMax = new int[len], colMax = new int[len];
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < len; j++) {
                rowMax[i] = Math.max(rowMax[i], grid[i][j]);
                colMax[j] = Math.max(colMax[j], grid[i][j]);
            }
        }

        int inc = 0;
        for(int i = 0; i < len; i++) {
            for(int j = 0; j < len; j++) {
                inc += Math.min(rowMax[i], colMax[j]) - grid[i][j];
            }
        }
        return inc;
    }

    public boolean splitArraySameAverage(int[] A) {
        int len = A.length;
        int sum = 0, max = Integer.MIN_VALUE;
        for(int num : A) {
            sum += num;
            max = Math.max(max, num);
        }
        if(max > sum / 2) return false;
        double avg = (double)sum / len;
        for(int size = 1; size <= A.length / 2; size++) {
            if(isInteger(avg * size) && xSum(A, size, 0, (int)(avg * size), 0)) {
                return true;
            }
        }
        return false;
    }

    private boolean isInteger(double num) {
        if(num - (int)num < 0.01) {
            return true;
        } else {
            return false;
        }
    }

    private boolean xSum(int[] A, int remain, int prevSum, int target, int startIdx) {
        if(remain == 0) {
            if(prevSum == target) {
                return true;
            } else {
                return false;
            }
        }

        for(int i = startIdx; i < A.length; i++) {
            if(xSum(A, remain - 1, prevSum + A[i], target, i + 1)) {
                return true;
            }
        }
        return false;
    }

    public String solution(String S) {
        // write your code in Java SE 8
        String prev = new String(S);
        while(true) {
            String after = prev.replace("AA", "");
            after = after.replace("BB", "");
            after = after.replace("CC", "");
            if(after.length() == prev.length()) {
                break;
            } else {
                prev = after;
            }
        }
        return prev;
    }

    public String solutionII(String S) {
        // write your code in Java SE 8
        StringBuilder sb = new StringBuilder();
        Stack<Character> stack = new Stack<>();

        if(S == null || S.length() == 0) return S;
        for(int i = 0; i < S.length(); i++) {
            char c = S.charAt(i);
            if(!stack.isEmpty() && stack.peek() == c) {
                stack.pop();
            } else {
                stack.push(c);
            }
        }
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        return sb.reverse().toString();

    }

    public int findClosestLeaf(TreeNode root, int k) {
        Map<TreeNode, TreeNode> parents = new HashMap<>();
        TreeNode target = new TreeNode(-1);
        dfs(parents, root, target, k);

        return bfs(target, parents).val;
    }

    private void dfs(Map<TreeNode, TreeNode> parents, TreeNode parent, TreeNode target, int k) {
        if(parent == null) {
            return;
        }

        if(parent.val == k) target = parent;

        if(parent.left != null) {
            parents.put(parent.left, parent);
            dfs(parents, parent.left, target, k);
        }
        if(parent.right != null) {
            parents.put(parent.right, parent);
            dfs(parents, parent.right, target, k);
        }
    }

    private TreeNode bfs(TreeNode start, Map<TreeNode, TreeNode> parents) {
        Queue<TreeNode> nodes = new LinkedList<>();
        nodes.add(start);
        //TreeNode res = null;
        while(!nodes.isEmpty()) {
            TreeNode head = nodes.poll();

            if(head.left == null && head.right == null) return head;

            TreeNode parent = null;
            if(parents.keySet().contains(head)) {
                parent = parents.get(head);
                nodes.add(parent);
            }
            if(head.left != null) {
                nodes.add(head.left);
            }
            if(head.right != null) {
                nodes.add(head.right);
            }
        }
        return null;
    }

    public String shortestCompletingWord(String licensePlate, String[] words) {
        Arrays.sort(words, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return o1.length() - o2.length();
            }
        });

        Map<Character, Integer> map = new HashMap<>();
        int count = 0;
        for(char c : licensePlate.toLowerCase().toCharArray()) {
            if(c >= 'a' && c <= 'z') {
                map.put(c, map.getOrDefault(c, 0 ) + 1);
                count++;
            }
        }

        for(String word : words) {
            if(check(map, word, count)) {
                return word;
            }
        }
        return null;
    }

    private boolean check(Map<Character, Integer> map, String word, int count) {
        int cnt = 0;
        Map<Character, Integer> copyMap = new HashMap<>(map);

        for(char c : word.toLowerCase().toCharArray()) {
            if(copyMap.containsKey(c) && copyMap.get(c) > 0) {
                cnt++;
                copyMap.put(c, copyMap.get(c) - 1);
            }
        }
        if(cnt == count) {
            return true;
        } else {
            return false;
        }
    }

    private final int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    public List<Integer> removeDuplicate(int[] array) {
        List<Integer> list = new ArrayList<>();
        list.add(array[0]);
        int prev = array[0];
        for(int i = 1; i < array.length; i++) {
            if(array[i] != prev) {
                list.add(array[i]);
                prev = array[i];
            }
        }
        return list;
    }

    public List<String> subdomainVisits(String[] cpdomains) {
        List<String> list = new ArrayList<>();
        if( cpdomains == null || cpdomains.length == 0) return list;
        Map<String, Integer> map = new HashMap<>();
        for(String cpdomain : cpdomains) {
            String[] info = cpdomain.split(" ");
            int times = Integer.parseInt(info[0]);
            List<String> domains = parseDomain(info[1]);
            for(String domain : domains) {
                map.put(domain, map.getOrDefault(domain, 0) + times);
            }
        }
        for(String key : map.keySet()) {
            list.add(map.get(key) + " " + key);
        }
        return list;
    }

    public int expressiveWords(String S, String[] words) {
        int res = 0;
        for(String word : words) {
            System.out.println(word +  "   " + canExtend(S, word));
//            if(canExtend(S, word)) {
//                res++;
//            }
        }
        return res;
    }

    private boolean canExtend(String s, String word) {
        if(s.length() < word.length()) return false;

        int row = word.length() + 1;
        int col = s.length() + 1;
        boolean[][] dp = new boolean[row][col];
        dp[0][0] = true;
        for(int i = 1; i < row; i++) {
            for(int j = 1; j < col; j++) {
                char cw = word.charAt(i - 1), cs = s.charAt(j - 1);
                if(cw == cs) {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                if(j >= 3 && cs == s.charAt(j - 2) && cs == s.charAt(j - 3)) {
                    int k = j - 3;
                    int l = i - 1;
                    while(k >= 1 && s.charAt(k) == s.charAt(k - 1)) k--;
                    while(l >= 1 && word.charAt(l) == word.charAt(l - 1)) l--;
                    dp[i][j] = dp[i][j] || dp[l + 1][k + 1];
                }
            }
        }
        return dp[row - 1][col - 1];
    }

    public double soupServings(int N) {
        Map<String, Double> cache = new HashMap<>();
        return soupDp(N, N, cache);
        //return cache.get(N + " " + N);
    }

//    private double soupsoupsoup(int remainA, int remainB) {
//        if(remainA <= 0 && remainB > 0) return 1;
//        if(remainA <= 0 && remainB <= 0) return 0.5;
//        if(remainA > 0 && remainB <= 0) return 0;
//        return 0.25 * soupsoupsoup(remainA - 100, remainB) +
//                0.25 * soupsoupsoup(remainA - 75, remainB - 25) +
//                0.25 * soupsoupsoup(remainA - 50, remainB - 50) +
//                0.25 * soupsoupsoup(remainA - 25, remainB - 75);
//    }

    private double soupDp(int remainA, int remainB, Map<String, Double> cache) {
        if(remainA <= 0 && remainB > 0) return 1;
        if(remainA <= 0 && remainB <= 0) return 0.5;
        if(remainA > 0 && remainB <= 0) return 0;
        // A > 0 B > 0
        if(cache.keySet().contains(remainA + " " + remainB)) {
            return cache.get(remainA + " " + remainB);
        } else {
            double res = 0.25 * soupDp(remainA - 100, remainB, cache) +
                    0.25 * soupDp(remainA - 75, remainB - 25, cache) +
                    0.25 * soupDp(remainA - 50, remainB - 50, cache) +
                    0.25 * soupDp(remainA - 25, remainB - 75, cache);

            return res;
        }
    }

    private List<String> parseDomain(String domain) {
        int len = domain.length();
        int dot = 0;
        List<String> list = new ArrayList<>();
        list.add(domain);
        for(int i = len - 1; i >= 0; i--) {
            if(dot == 2) {
                break;
            }
            if(domain.charAt(i) == '.') {
                list.add(domain.substring(i + 1));
                dot++;
            }
        }
        return list;
    }

    public boolean xorGame(int[] nums) {
        return false;
    }

    public boolean circularArrayLoop(int[] nums) {
        if(nums == null || nums.length <= 1) return false;

        int len = nums.length;

        for(int i = 0; i < nums.length; i++) {
            if(nums[i] == 0) continue;

            int slow = i, fast = (i + nums[i] + len) % len;
            while(slow != fast && nums[slow] * nums[fast] > 0) {
                slow = (slow + nums[slow] + len) % len;
                fast = (fast + nums[fast] + len) % len;
                fast = (fast + nums[fast] + len) % len;
            }
            if(nums[slow] * nums[fast] <= 0) {
                int tmp = i;
                int head = nums[i];
                while(head * nums[tmp] > 0) {
                    tmp = (tmp + nums[tmp] + len) % len;
                    nums[tmp] = 0;
                }
            } else {
                return true;
            }
        }
        return false;
    }

    public List<Integer> topKFrequent(int[] nums, int k) {
        int len = nums.length;
        Map<Integer, Integer> numToTimes = new HashMap<>();

        for(int i = 0; i < len; i++) {
            int num = nums[i];
            numToTimes.put(num, numToTimes.getOrDefault(num, 0) + 1);
        }

        List<Map.Entry<Integer, Integer>> list = new ArrayList<>();
        for(Map.Entry<Integer, Integer> entry : numToTimes.entrySet()) {
            list.add(entry);
        }

        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {
            @Override
            public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
                return o2.getValue() - o1.getValue();
            }
        });

        List<Integer> res = new ArrayList<>();
        for(int i = 0; i < k; i++) {
            res.add(list.get(i).getKey());
        }
        return res;
    }

    public int countCornerRectangles(int[][] grid) {
        Map<String, Integer> map = new HashMap<>();
        int row = grid.length, col = grid[0].length;
        for(int i = 0; i < row; i++) {
            for(int start = 0; start < col; start++) {
                for(int end = start + 1; end < col; end++) {
                    if(grid[i][start] == 1 && grid[i][end] == 1) {
                        map.put(start + " " + end, map.getOrDefault(start + " " + end, 0) + 1);
                    }
                }
            }
        }
        int res = 0;
        for(String key : map.keySet()) {
            int num = map.get(key);
            res += num * (num - 1) / 2;
        }
        return res;
    }

    public List<String> ipToCIDR(String ip, int n) {
        List<String> res = new ArrayList<>();
        for(int i = 0; i < n;) {
            int tailingZero = tailingZeros(ip);
            int range = (int)Math.pow(2, tailingZero);
            if(i + range < n) {
                res.add(ip + "/" + (32 - tailingZero));
                i += range;
                ip = updateIP(ip, range);
            } else {
                int subRange = range;
                int shift = 0;
                while(i + subRange > n) {
                    subRange = subRange / 2;
                    shift++;
                }
                res.add(ip + "/" + (32 - tailingZero + shift));
                ip = updateIP(ip, subRange);
                i = i + subRange;
            }
        }
        return res;
    }

    private String updateIP(String ip, int range) {
        String[] str = ip.split("\\.");
        int add = range;
        for(int i = 3; i >= 0; i--) {
            int last = Integer.parseInt(str[i]) + add;
            if(last >= 256){
                last = last % 256;
                add = last / 256;
                str[i] = last + "";
            } else {
                str[i] = last + "";
                break;
            }

        }

        return str[0] + "." + str[1] + "." + str[2] + "." + str[3];
    }

    private int tailingZeros(String ip) {
        String[] str = ip.split("\\.");
        int last = Integer.parseInt(str[3]);
        if(last == 0) {
            int tmp = Integer.parseInt(str[2]);
            if(tmp % 4 == 0) {
                return 10;
            } else if(tmp % 2 == 0) {
                return 9;
            } else {
                return 8;
            }
        } else {
            int res = 0;
            while(last % 2 == 0) {
                res++;
                last = last / 2;
            }
            return res;
        }
    }

    public int openLock(String[] deadends, String target) {
        Set<String> deadlocks = new HashSet<>(Arrays.asList(deadends));
        if(deadlocks.contains("0000")) return -1;
        int step = 1;
        Queue<String> q = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        q.add("0000");
        visited.add("0000");
        while (!q.isEmpty()) {
            int size = q.size();
            for(int i = 0; i < size; i++) {
                String cur = q.poll();
                for(int pos = 0; pos < 4; pos++) {
                    int num = Integer.parseInt(cur.charAt(pos) + "");
                    StringBuilder sb = new StringBuilder(cur);
                    sb.replace(pos, pos + 1, (num - 1 + 10) % 10 + "");
                    String first = sb.toString();
                    if(first.equals(target)) {
                        return step;
                    } else if(!deadlocks.contains(first) && !visited.contains(first)) {
                        q.add(first);
                        visited.add(first);
                    }
                    sb.replace(pos, pos + 1, (num + 1 + 10) % 10 + "");
                    String second = sb.toString();
                    if(second.equals(target)) {
                        return step;
                    } else if(!deadlocks.contains(second) && !visited.contains(second)) {
                        q.add(second);
                        visited.add(second);
                    }
                }
            }
            step++;
        }
        return -1;
    }

    public int reachNumber(int target) {
        target = Math.abs(target);
        int total = 0;
        int step = 0;
        while(total < target) {
            step++;
            total += step;
        }
        while((total - target) % 2 == 1) {
            step++;
            total += step;
        }
        return step;
    }

    public int[] pourWater(int[] heights, int V, int K) {
        for(int i = 0; i < V; i++) {
            boolean left = (K == 0 || heights[K] < heights[K - 1]);
            boolean right = (K == heights.length - 1 || heights[K] < heights[K + 1]);
            if(left && right) {
                heights[K]++;
            } else {
                int posl = flowToLeft(heights, K);
                if(posl == K) {
                    int posr = flowToRight(heights, K);
                    heights[posr]++;
                } else {
                    heights[posl]++;
                }
            }
        }
        return heights;
    }

    private int flowToLeft(int[] heights, int pos) {
        int finalPos = pos, potential = pos - 1;
        while(potential >= 0) {
            if(heights[potential] == heights[potential + 1]) {

            } else if(heights[potential] > heights[potential + 1]) {
                break;
            } else {
                finalPos = potential;
            }
            potential--;
        }
        return finalPos;
    }

    private int flowToRight(int[] heights, int pos) {
        int finalPos = pos, potential = pos + 1;
        while (potential < heights.length) {
            if(heights[potential] == heights[potential - 1]) {

            } else if(heights[potential] > heights[potential - 1]) {
                break;
            } else {
                finalPos = potential;
            }
            potential++;
        }
        return finalPos;
    }

    public boolean pyramidTransition(String bottom, List<String> allowed) {
        Map<String, List<Character>> map = new HashMap<>();
        for(String str : allowed) {
            String key = str.substring(0, 2);
            char val = str.charAt(2);
            if(!map.containsKey(key)) {
               map.put(key, new ArrayList<>());
            }
            map.get(key).add(val);
        }

        return buildPyramid(bottom, map, 0, new char[bottom.length() - 1]);
    }

    private boolean buildPyramid(String prev,  Map<String, List<Character>> allowed, int pos, char[] chars) {
        if(prev.length() == 2) {
            if(allowed.containsKey(prev)) {
                return true;
            } else {
                return false;
            }
        }

        String key = prev.substring(pos, pos + 2);
        if(allowed.containsKey(key)) {
            for(char val : allowed.get(key)) {
                chars[pos] = val;
                if(pos == prev.length() - 2) {
                    boolean next = buildPyramid(new String(chars), allowed, 0, new char[chars.length - 1]);
                    if(next) {
                        return true;
                    }
                } else {
                    boolean next = buildPyramid(prev, allowed, pos + 1, chars);
                    if(next) {
                        return true;
                    }
                }
            }
            return false;
        } else {
            return false;
        }
    }

    public int intersectionSizeTwo(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if(o1[1] != o2[1]) {
                    return o1[1] - o2[1];
                } else {
                    return o2[0] - o1[0];
                }
            }
        });

        int largest = intervals[0][1], second = intervals[0][1] - 1;
        int res = 2;
        for(int i = 1; i < intervals.length; i++) {
            int[] interval = intervals[i];

            if(interval[0] <= second) {
                continue;
            } else if(interval[0] > second && interval[0] <= largest) {
                second = largest;
                largest = interval[1];
                res++;
            } else {
                // interval[0] > largest
                second = interval[1] -1;
                largest = second + 1;
                res += 2;
            }
        }
        return res;
    }

    public String boldWords(String[] words, String S) {
        int[] tag = new int[S.length()];
        for(String word : words) {
            int idx = 0;
            while (true) {
                idx = S.indexOf(word, idx);
                if (idx != -1) {
                    for (int i = 0; i < word.length(); i++) {
                        tag[i + idx] = 1;
                    }
                    idx++;
                } else {
                    break;
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < tag.length; i++) {
            if((i == 0 && tag[i] == 1) || (tag[i] == 1 && tag[i - 1] == 0)) {
               sb.append("<b>").append(S.charAt(i));
               continue;
            }
            if(i > 0 && tag[i] == 0 && tag[i - 1] == 1) {
                sb.append("</b>").append(S.charAt(i));
                continue;
            }
            sb.append(S.charAt(i));
        }
        if(tag[tag.length - 1] == 1) sb.append("</b>");
        return sb.toString();
    }

    public List<Interval> employeeFreeTime(List<List<Interval>> schedule) {
        List<Interval> occupy = new ArrayList<>();
        for(List<Interval> list : schedule) {
            occupy.addAll(list);
        }
        occupy.sort(new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start - o2.start;
            }
        });
        List<Interval> res = new ArrayList<>();
        Interval prev = occupy.get(0);
        for(int i = 1; i < occupy.size(); i++) {
            Interval cur = occupy.get(i);
            if(cur.start > prev.end) {
                res.add(new Interval(prev.end, cur.start));
            }
            prev.end = Math.max(prev.end, cur.end);
        }
        return res;
    }

    public int[] anagramMappings(int[] A, int[] B) {
        Map<Integer, List<Integer>> numToIdx = new HashMap<>();
        for(int i = 0; i < B.length; i++) {
            if(!numToIdx.containsKey(B[i])) {
                numToIdx.put(B[i], new ArrayList<>());
            }
            numToIdx.get(B[i]).add(i);
        }
        int[]res = new int[A.length];
        for(int i = 0; i < A.length; i++) {
            int a = A[i];
            List<Integer> list = numToIdx.get(a);
            res[i] = list.remove(0);
        }
        return res;
    }

    public String makeLargestSpecial(String s) {
        if(s.length() == 0) return s;
        List<String> list = new ArrayList<>();
        int cnt = 0, last = 0;
        for(int i = 0; i < s.length(); i++) {
            cnt += (s.charAt(i) == '1' ? 1 : -1);
            if(cnt == 0) {
                list.add("1" + makeLargestSpecial(s.substring(last + 1, i)) + "0");
                last = i + 1;
            }
        }
        Collections.sort(list, Collections.reverseOrder());
        StringBuilder sb = new StringBuilder();
        for(String ss : list) sb.append(ss);
        return sb.toString();
    }

    public int minSwapsCouples(int[] row) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < row.length; i++) {
            map.put(row[i], i);
        }
        int swap = 0;
        for(int i = 1; i < row.length; i = i + 2) {
            if((row[i] % 2 == 1 && row[i - 1] == row[i] - 1) || (row[i] % 2 == 0 && row[i - 1] == row[i] + 1)) {
                continue;
            } else {
                int idx = -1;
                idx = map.get(row[i - 1] + (row[i - 1] % 2 == 0 ? 1 : -1));
                int tmp = row[i];
                row[i] = row[idx];
                row[idx] = tmp;
                map.put(tmp, idx);
                swap++;
            }
        }
        return swap;
    }

    public double largestTriangleArea(int[][] points) {
        double res = 0.0;
        int len = points.length;
        for(int i = 0; i < len; i++) {
            for(int j = i + 1; j < len; j++) {
                for(int k = j + 1; k < len; k++) {
                    int x1 = points[i][0], y1 = points[i][1];
                    int x2 = points[j][0], y2 = points[j][1];
                    int x3 = points[k][0], y3 = points[k][1];
                    double area = Math.abs(0.5 * (x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3));
                    //System.out.println("area = " + area);
                    if(area > res) {
                        res = area;
                    }
                }
            }
        }
        return res;
    }

    public TreeNode pruneTree(TreeNode root) {
        if(root == null) return root;
        int sum = treeAdd(root);

        if(root.val == 0) return null;

        if(root.left != null && root.left.val == 0) {
            root.left = null;
        } else {
            root.left = pruneTree(root.left);
            root.left.val = root.left.val % 10;
        }
        if(root.right != null && root.right.val == 0) {
            root.right = null;
        } else {
            root.right = pruneTree(root.right);
            root.right.val = root.right.val % 10;
        }
        return root;
    }

    private int treeAdd(TreeNode root) {
        if(root == null) {
            return 0;
        }
        int left = treeAdd(root.left);//root.left
        int right = treeAdd(root.right);
        root.val += (left + right) * 10;
        return root.val;
    }

    class Element{
        double sum;
        int times;
        public Element(double sum, int times) {
            this.sum = sum;
            this.times = times;
        }
    }

    public double largestSumOfAverages(int[] A, int K) {
        double[] sum = new double[A.length];
        for(int i = 0; i < A.length; i++) {
            sum[i] = A[i] + (i == 0 ? 0 : sum[i - 1]);
        }

        double[][] dp = new double[K][A.length];
        for(int j = 0; j < A.length; j++) {
            dp[0][j] = sum[j] / (j + 1);
        }
        for(int i = 1; i < K; i++) {
            for(int j = i; j < A.length; j++) {
                // treat the last element as a separated element
                dp[i][j] = dp[i - 1][j - 1] + A[j];
                for(int l = j - 1; l >= i; l--) {
                    double avg = (sum[j] - sum[l - 1]) / (j - l + 1);
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][l - 1] + avg);
                }
            }
        }
        return dp[K - 1][A.length - 1];
    }


    public int numBusesToDestination(int[][] routes, int S, int T) {
        if(S == T) return 0;
        Map<Integer, List<Integer>> map = new HashMap<>();
        for(int i = 0; i < routes.length; i++) {
            int[] route = routes[i];
            for(int station : route) {
                if(!map.containsKey(station)) map.put(station, new ArrayList<>());
                map.get(station).add(i);
            }
        }

        Queue<Integer> q = new LinkedList<>();
        q.add(S);
        Set<Integer> visited = new HashSet<>();

        int step = 1;

        while (!q.isEmpty()) {
            int size = q.size();
            for(int i = 0; i < size; i++) {
                int station = q.poll();
                List<Integer> buses = map.get(station);
                for(int bus : buses) {
                    if(!visited.add(bus)) continue;

                    for(int ring : routes[bus]) {
                        if(ring == T) return step;
                        if(ring != station) q.add(ring);
                    }
                }
            }
            step++;
        }
        return -1;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        if(k == 1 || head == null || head.next == null) return head;

        ListNode dummy = new ListNode(-1), p = dummy;
        dummy.next = head;
        while (p.next != null) {
            int step = 0;
            ListNode start = p;
            while(step++ < k) {
                if(p.next != null) {
                    p = p.next;
                } else {
                    return dummy.next;
                }
            }
            p =  reverse(start, p);
        }
        return dummy.next;
    }

    private ListNode reverse(ListNode start, ListNode end) {
        ListNode endNext = end.next;
        ListNode p = start.next, res = start.next;
        ListNode next = p.next;
        while(p != end) {
            ListNode nn = next.next;
            next.next = p;
            p = next;
            next = nn;
        }
        start.next.next = next;
        start.next = p;
        return res;
    }

    public int countPrimeSetBits(int L, int R) {
        Set<Integer> prime = new HashSet<>();
        prime.addAll(Arrays.asList(2,3,5,7,11,13,17,19,23,29,31));
        int res = 0;
        for(int num = L; num <= R; num++) {
            int count = countOne(num);
            if(prime.contains(count)) {
                res++;
            }
        }
        return res;
    }

    private int countOne(int num) {
        int ones = 0;
        while(num != 0) {
            ones += (num % 2 == 1 ? 1 : 0);
            num = num / 2;
        }
        return ones;
    }

    public List<Integer> partitionLabels(String S) {
        Map<Character, Integer> first = new HashMap<>();
        Map<Character, Integer> last = new HashMap<>();

        for(int i = 0; i < S.length(); i++) {
            last.put(S.charAt(i), i);
        }

        List<Integer> res = new ArrayList<>();
        int lastIdx = 0, prev = 0;
        for(int i = 0; i < S.length(); i++) {
            char c = S.charAt(i);
            lastIdx = Math.max(lastIdx, last.get(c));
            if(i == lastIdx) {
                res.add(lastIdx - prev + 1);
                prev = lastIdx + 1;
            }
        }
        return res;
    }

    public String reorganizeString(String S) {
        int[] cnt = new int[26];
        int max = 0;
        for(int i = 0; i < S.length(); i++) {
            int tmp = ++cnt[S.charAt(i) - 'a'];
            max = Math.max(tmp, max);
        }

        if(max <= (S.length() + 1) / 2) {
            StringBuilder sb = new StringBuilder();
            Queue<Character> q = new PriorityQueue<>(new Comparator<Character>() {
                @Override
                public int compare(Character o1, Character o2) {
                    return cnt[o2 - 'a'] - cnt[o1 - 'a'];
                }
            });
            for(int i = 0; i < 26; i++) {
                if(cnt[i] != 0) {
                    q.add((char)('a' + i));
                }
            }
            char prev = 'A';
            for(int i = 0; i < S.length(); i++) {
                char first = q.poll();
                if(first != prev) {
                    sb.append(first);
                    cnt[first - 'a']--;
                    prev = first;
                } else {
                    char second = q.poll();
                    sb.append(second);
                    cnt[second - 'a']--;
                    prev = second;
                    q.add(second);
                }
                q.add(first);
            }
            return sb.toString();
        } else {
            return "";
        }
    }

    public int maxChunksToSorted(int[] arr) {
        int[] minArr = new int[arr.length];
        minArr[arr.length - 1] = Integer.MAX_VALUE;;
        for(int i = arr.length - 2; i >= 0; i--) {
            minArr[i] = Math.min(arr[i + 1],  minArr[i + 1]);
        }
        int res = 0, max = 0;
        for(int i = 0; i < arr.length; i++) {
            max = Math.max(arr[i], max);
            if(max <= minArr[i]) {
                res++;
                max = 0;
            }
        }
        return res;
    }

    public int swimInWater(int[][] grid) {
        int time = grid[0][0];
        int[][] dir = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Queue<int[]> pq = new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return grid[o1[0]][o1[1]] - grid[o2[0]][o2[1]];
            }
        });
        pq.add(new int[]{0, 0});
        Set<String> visited = new HashSet<>();
        visited.add("0 0");
        while(!pq.isEmpty()) {
            int[] pos = pq.poll();
            if(time < grid[pos[0]][pos[1]]) {
                time = grid[pos[0]][pos[1]];
            }
            if(pos[0] == grid.length - 1 && pos[1] == grid[0].length - 1) break;
            for(int i = 0; i < 4; i++) {
                int x = pos[0] + dir[i][0], y = pos[1] + dir[i][1];
                if(x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && visited.add(x + " " + y)) {
                    pq.add(new int[]{x, y});
                }
            }
        }
        return time;
    }

    public int movesToChessboard(int[][] board) {
        if(board == null || board.length == 0) return 0;
        int len = board.length;
        // each row is completely the same or opposite with the first row
        for(int i = 1; i < len; i++) {
            boolean flag = board[i][0] == board[0][0];
            for(int j = 1; j < len; j++) {
                if((board[i][j] == board[0][j]) != flag) {
                    return -1;
                }
            }
        }
        int rowCnt = 0, colCnt = 0;
        for(int i = 0; i < len; i++) rowCnt += board[0][i] == 1 ? 1 : -1;
        for(int i = 0; i < len; i++) colCnt += board[i][0] == 1 ? 1 : -1;
        if(rowCnt <= -2 || rowCnt >= 2 || colCnt <= -2 || colCnt >= 2) return -1;

        int row = 0, col = 0;
        for(int i = 0; i < len; i++) {
            if((i % 2 == 1) != (board[i][0] != board[0][0])) {
                row++;
            }
        }
        for(int j = 0; j < len; j++) {
            if((j % 2 == 1) != (board[0][0] != board[0][j])) {
                col++;
            }
        }

        System.out.println(row + "  " + col + "  len = " + len);
        if(len % 2 == 0) {
            return Math.min(row, len - row) / 2 + Math.min(col, len - col) / 2;
        } else {
            return (row % 2 == 1 ? (len - row) / 2 : row / 2) +
                    (col % 2 == 1 ? (len - col) / 2 : col / 2);
        }

        //return (row % 2 == 0 ? row / 2 : (len - row) / 2) +(col % 2 == 0 ? col / 2 : (len - col) / 2);
    }

    public int orderOfLargestPlusSign(int N, int[][] mines) {
        int[][] grid = new int[N][N];
        for(int i = 0; i < N; i++) {
            Arrays.fill(grid[i], 1);
        }
        for(int[] pos : mines) {
            grid[pos[0]][pos[1]] = 0;
        }
        int[][] u = new int[N][N], d = new int[N][N], l = new int[N][N], r = new int[N][N];
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(i == 0) {
                    u[i][j] = grid[i][j];
                } else {
                    u[i][j] = grid[i][j] == 0 ? 0 : u[i - 1][j] + 1;
                }
            }
        }
        for(int i = N - 1; i >= 0; i--) {
            for(int j = 0; j < N; j++) {
                if(i == N - 1) {
                    d[i][j] = grid[i][j];
                } else {
                    d[i][j] = grid[i][j] != 0 ? d[i + 1][j] + 1 : 0;
                }
            }
        }
        for(int j = 0; j < N; j++) {
            for(int i = 0; i < N; i++) {
                if(j == 0) {
                    l[i][j] = grid[i][j];
                } else {
                    l[i][j] = grid[i][j] != 0 ? l[i][j - 1] + 1 : 0;
                }
            }
        }
        for(int j = N - 1; j >= 0; j--) {
            for(int i = 0; i < N; i++) {
                if(j == N - 1) {
                    r[i][j] = grid[i][j];
                } else {
                    r[i][j] = grid[i][j] != 0 ? r[i][j + 1] + 1 : 0;
                }
            }
        }
        int max = 0;
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                max = Math.max(max, Math.min(Math.min(u[i][j], d[i][j]), Math.min(l[i][j], r[i][j])));
            }
        }
        return max;
    }

    class TrieNode {
        boolean root = false;
        TrieNode[] children = new TrieNode[26];
    }

    TrieNode root = new TrieNode();
    public String replaceWords(List<String> dict, String sentence) {
        for(int i = 0 ; i < dict.size(); i++) {
            TrieNode p = root;
            String word = dict.get(i);
            for(int j = 0; j < word.length(); j++) {
                char c = word.charAt(j);
                if(p.children[c - 'a'] == null) {
                    p.children[c - 'a'] = new TrieNode();
                }
                p = p.children[c - 'a'];
                if(p.root) break;
            }
            p.root = true;
        }

        String[] words = sentence.split(" ");
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < words.length; i++) {
            String replace = find(words[i]);
            if(replace != null) {
                sb.append(replace);
            } else {
                sb.append(words[i]);
            }
            if(i != words.length - 1) sb.append(" ");
        }
        return sb.toString();
    }

    private String find(String word) {
        TrieNode p = root;
        for(int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if(p.children[c - 'a'] == null) {
                return null;
            } else {
                p = p.children[c - 'a'];
                if(p.root) {
                    return word.substring(0, i + 1);
                }
            }
        }
        return null;
    }
    //"bar foo foo bar the foo barman"
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new ArrayList<>();
        int len = words[0].length();
        for(int offset = 0; offset < len; offset++) {
            Map<String, Integer> map = new HashMap<>();
            for(String word : words) map.put(word, map.getOrDefault(word, 0) + 1);

            List<String> list = new ArrayList<>();
            for(int i = 0 + offset; i + len <= s.length(); i = i + len) {
                list.add(s.substring(i, i + len));
            }
            int cnt = words.length;
            int from = 0, to = 0;
            while (to < list.size()) {
                String tword = list.get(to++);
                if(map.containsKey(tword)) {
                    map.put(tword, map.get(tword) - 1);
                    if(map.get(tword) >= 0) cnt--;
                }
                while(cnt == 0) { //from < to - cnt
                    if(cnt == 0 && from == to - words.length) res.add(offset + from * len);
                    String fword = list.get(from++);
                    if(map.containsKey(fword)) {
                        map.put(fword, map.get(fword) + 1);
                    }
                    if(map.get(fword) > 0) cnt++;
                }
            }
        }
        return res;
    }

    public boolean isScramble(String s1, String s2) {
        if(s1.equals(s2)) {
            return true;
        } else {
            if(s1.length() != s2.length()) {
                return false;
            } else {
                for(int i = 1; i < s1.length(); i++) {
                    if(isScramble(s1.substring(0, i), s2.substring(0, i))
                            && isScramble(s1.substring(i), s2.substring(i))) {
                        return true;
                    }
                    if(isScramble(s1.substring(0, i), s2.substring(s2.length() - i))
                            && isScramble(s1.substring(i), s2.substring(0, s2.length() - i))) {
                        return true;
                    }
                }
                return false;
            }
        }
    }

    public int removeBoxes(int[] boxes) {
        int len = boxes.length;
        int[][][] dp = new int[len][len][len];
        return removeBoxes(boxes, 0, len - 1, 0, dp);
    }

    private int removeBoxes(int[] boxes, int i, int j, int k, int[][][] dp) {
        if(i > j) return 0;
        if(dp[i][j][k] != 0) {
            return dp[i][j][k];
        }
        for(; i + 1 <= j && boxes[i] == boxes[i + 1]; i++, k++);
        int res = (k + 1) * (k + 1) + removeBoxes(boxes, i + 1, j, 0, dp);
        for(int m = i + 1; m <= j; m++) {
            if(boxes[m] == boxes[i]) {
                res = Math.max(res, removeBoxes(boxes, i + 1, m -1, 0, dp) +
                        removeBoxes(boxes, m, j, k + 1, dp));
            }
        }
        dp[i][j][k] = res;
        return res;
    }

    public int maxCoins(int[] nums) {
        int len = nums.length;
        // left ~ [-1, len - 1], right ~ [0, len]
        int[][] dp = new int[len + 1][len + 1];
        return maxCoins(nums, dp, 0, len - 1, -1, len);
    }

    // i, j is index in nums
    private int maxCoins(int[] nums, int[][] dp, int i, int j, int left, int right) {
        if(i > j) return 0;
        if(i == j) {
            int l = left == -1 ? 1 : nums[left], r = right == nums.length ? 1 : nums[right];
            int res = nums[i] * l * r;
            dp[left + 1][right] = res;
            return res;
        }

        //System.out.println((left + 1) + " " + i + " " + j + " " + right);
        if(dp[left + 1][right] != 0) return dp[left + 1][right];
        int res = maxCoins(nums, dp, i + 1, j, i, right) +
                nums[i] * (left == -1 ? 1 : nums[left]) * (right == nums.length ? 1 : nums[right]);

        for(int k = i + 1; k <= j; k++) {
            int subres = maxCoins(nums, dp, i, k - 1, left, k) + maxCoins(nums, dp, k + 1, j, k, right)
                    + nums[k] * (left == -1 ? 1 : nums[left]) * (right == nums.length ? 1 : nums[right]);
            res = Math.max(res, subres);
        }
        dp[left + 1][right] = res;
        return res;
    }

    public int numComponents(ListNode head, int[] G) {
        Set<Integer> set = new HashSet<>();
        for(int num : G) set.add(num);

        boolean connected = false;
        ListNode p = head;
        int res = 0;
        while(p != null) {
            if(set.contains(p.val)) {
                if(connected) {

                } else {
                    connected = true;
                }
                if(p.next == null) {
                    res++;
                }
            } else {
                if(connected) {
                    res++;
                    connected = false;
                }
            }
            p = p.next;
        }
        return res;
    }

    public List<String> ambiguousCoordinates(String S) {
        List<List<String>> all = new ArrayList<>();
        List<String> prev = new ArrayList<>();
        cut(all, S.substring(1, S.length() - 1), 0, 3, prev);
        List<String> res = new ArrayList<>();
        for(List<String> list : all) {
            int size = list.size();
            if(size == 2) {
                if(validInteger(list.get(0)) && validInteger(list.get(1))) {
                    res.add("(" + list.get(0) + ", " + list.get(1) + ")");
                }
            } else if(size == 3) {
                if(validInteger(list.get(0)) && validDecimal(list.get(1)) && validInteger(list.get(2)))  {
                    res.add("(" + list.get(0) + "." + list.get(1) + ", " + list.get(2) + ")");
                }
                if(validInteger(list.get(0)) && validInteger(list.get(1)) && validDecimal(list.get(2)))  {
                    res.add("(" + list.get(0) + ", " + list.get(1) + "." + list.get(2) + ")");
                }
            } else {
                if(validInteger(list.get(0)) && validDecimal(list.get(1))
                        && validInteger(list.get(2)) && validDecimal(list.get(3))) {
                    res.add("(" + list.get(0) + "." + list.get(1) + ", " + list.get(2) + "." + list.get(3) + ")");
                }
            }
        }
        return res;
    }

    private void cut(List<List<String>> all, String str, int start, int remain, List<String> prev) {
        if(start == str.length()) {
            all.add(new ArrayList<>(prev));
            return;
        }

        for(int i = start; i + 1 < str.length(); i++) {
            String sub1 = str.substring(start, i + 1), sub2 = str.substring(i + 1);
            prev.add(sub1);
            List<String> subList = new ArrayList<>(prev);
            subList.add(sub2);
            all.add(subList);
            if(remain > 1) {
                cut(all, str, i + 1, remain - 1, prev);
            }
            prev.remove(prev.size() - 1);
        }
    }

    private boolean validInteger(String str) {
        return str.equals("0") || !str.startsWith("0");
    }

    private boolean validDecimal(String str) {
        return !str.endsWith("0");
    }

    public int minCut(String s) {
        int len = s.length();
        boolean[][] pal = new boolean[len][len];
        int[] cut = new int[len];
        for(int j = 0; j < len; j++) {
            int min = j;
            for(int i = 0; i <= j; i++) {
                if(s.charAt(i) == s.charAt(j) && (i + 1 > j - 1 || pal[i + 1][j - 1])) {
                    pal[i][j] = true;
                    min = Math.min(min, i == 0 ? 0 : cut[i - 1] + 1);
                }
            }
            cut[j] = min;
        }
        return cut[len - 1];
    }

    public int calculateMinimumHP(int[][] dungeon) {
        int row = dungeon.length, col = dungeon[0].length;
        int[][] dp = new int[row + 1][col + 1];
        for(int i = 0; i <= row; i++) {
            dp[i][col] = Integer.MAX_VALUE;
        }
        for(int j = 0; j <= col; j++) {
            dp[row][j] = Integer.MAX_VALUE;
        }
        dp[row][col - 1] = 0; dp[row - 1][col] = 0;
        for(int i = row - 1; i >= 0; i--) {
            for(int j = col - 1; j >= 0; j--) {
                dp[i][j] = Math.max(Math.min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j], 0);
            }
        }
        return dp[0][0];
    }

    public int cherryPickup(int[][] grid) {
        int len = grid.length;
        int[][][] dp = new int[len + 1][len + 1][len + 1];

        for(int i = 0; i <= len; i++) {
            for(int j = 0; j <= len; j++) {
                Arrays.fill(dp[i][j], -1);
            }
        }

        dp[1][1][1] = grid[0][0];

        for(int x1 = 0; x1 < len; x1++) {
            for(int y1 = 0; y1 < len; y1++) {
                for(int x2 = 0; x2 < len; x2++) {
                    int y2 = x1 + y1 - x2;
                    if(dp[x1 + 1][y1 + 1][x2 + 1] >=0 || y2 < 0 || y2 >= len
                            || grid[x1][y1] < 0 || grid[x2][y2] < 0) continue;
                    int tmp1 = Math.max(dp[x1][y1 + 1][x2 + 1], dp[x1 + 1][y1][x2 + 1]);
                    int tmp2 = Math.max(dp[x1][y1 + 1][x2], dp[x1 + 1][y1][x2]);
                    int res = Math.max(tmp1, tmp2);
                    if(res != -1) {
                        dp[x1 + 1][y1 + 1][x2 + 1] = res + grid[x1][y1];
                      if(x1 != x2) dp[x1 + 1][y1 + 1][x2 + 1] += grid[x2][y2];
                    }
                }
            }
        }
        return Math.max(0, dp[len][len][len]);
    }

    public String removeDuplicateLetters(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for(int i = 0; i < s.length(); i++) {
            map.put(s.charAt(i), i);
        }
        char[] res = new char[map.size()];
        int start = 0, end = getMinIndex(map);
        for(int i = 0; i < res.length; i++) {
            char minChar = 'z' + 1;
            for(int j = start; j <= end; j++) {
                if(map.containsKey(s.charAt(j)) && s.charAt(j) < minChar) {
                    minChar = s.charAt(j);
                    start = j + 1;
                }
            }
            map.remove(minChar);
            if(s.charAt(end) == minChar) {
                end = getMinIndex(map);
            }
            res[i] = minChar;
        }
        return new String(res);
    }

    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int n = nums1.length, m = nums2.length;
        int[] res = new int[k];
        for(int i = Math.max(0, k - m); i <= n && i <= k; i++) {
            int[] candidate = merge(maxNumber(nums1, i), maxNumber(nums2, k - i ));
            if(larger(candidate, 0, res, 0)) {
                res = candidate;
            }
        }
        return res;
    }

    private int[] maxNumber(int[] nums, int len) {
        if (len == 0) return new int[0];

        int[] res = new int[len];
        // i for fast pointer, j for pointer of res
        for(int i = 0, j = 0; i < nums.length; i++) {
            while(nums.length - i + j > len && j > 0 && nums[i] > res[j - 1]) {
                j--;
            }
            if(j < len) res[j++] = nums[i];
        }
        return res;
    }

    private int[] merge(int[] nums1, int[] nums2) {
        int n = nums1.length, m = nums2.length;
        int[] res = new int[n + m];
        int i = 0, j = 0, k = 0;
        while(i < n && j < m) {
            res[k++] = larger(nums1, i, nums2, j) ? nums1[i++] : nums2[j++];
        }
        while(i < n) res[k++] = nums1[i++];
        while(j < m) res[k++] = nums2[j++];
        return res;
    }

    private boolean larger (int[] nums1, int i, int[] nums2, int j) {
        while(i < nums1.length && j < nums2.length && nums1[i] == nums2[j]) {
            i++;
            j++;
        }
        return j == nums2.length || (i < nums1.length && nums1[i] > nums2[j]);
    }

    private int getMinIndex(Map<Character, Integer> map) {
        int res = Integer.MAX_VALUE;
        for(int val : map.values()) {
            res = Math.min(val, res);
        }
        return res;
    }

    public List<Integer> countSmaller(int[] nums) {
        BSTNode root = null;
        // we want to use asList() method
        Integer[] res = new Integer[nums.length];
        for(int i = nums.length - 1; i >= 0; i--) {
            root = countSmaller(root, 0, nums[i], res, i);
        }
        return Arrays.asList(res);
    }

    private BSTNode countSmaller(BSTNode node, int preSmaller, int val, Integer[] res, int idx) {
        if(node == null) {
            res[idx] = preSmaller;
            return new BSTNode(val);
        }
        if(val == node.val) {
            node.dup++;
            res[idx] = preSmaller;
            return node;
        } else if(val < node.val) {
            node.smaller++;
            node.left = countSmaller(node.left, preSmaller, val, res, idx);
        } else {
            // val > node.val
            node.right = countSmaller(node.right, preSmaller + node.dup + node.smaller, val, res, idx);
        }
        return node;
    }

//    class BSTNode {
//        long val;
//        int smaller;
//        int bigger;
//        int dup;
//        BSTNode left = null, right = null;
//        public BSTNode (long val){
//            this.val = val;
//            this.smaller = 0;
//            this.bigger = 0;
//            this.dup = 1;
//        }
//    }

    public int countRangeSum(int[] nums, int lower, int upper) {
        long[] prefixSum = new long[nums.length + 1];
        long sum = 0;
        for(int i = 0; i < prefixSum.length; i++) {
            prefixSum[i] = sum;
            sum += ((i == prefixSum.length - 1) ? 0 : nums[i]);
        }
        BSTNode root = null;
        int res = 0;
        for(int i = 0; i < prefixSum.length; i++) {
            int s = smaller(root, prefixSum[i] - upper);
            int b = bigger(root, prefixSum[i] - lower);
            res += Math.max(0, i - s - b);
            root = addNode(root, prefixSum[i]);
        }
        return res;
    }

    private int smaller(BSTNode node, long val) {
        if(node == null) return 0;
        if(node.val == val) {
            return node.smaller ;
        } else if(node.val < val) {
            return node.dup + node.smaller + smaller(node.right, val);
        } else {
            return smaller(node.left, val);
        }
    }

    private int bigger(BSTNode node, long val) {
        if(node == null) return 0;
        if(node.val == val) {
            return node.bigger;
        } else if(node.val < val) {
            return bigger(node.right, val);
        } else {
            return bigger(node.left, val) + node.bigger + node.dup;
        }
    }

    private BSTNode addNode(BSTNode root, long val) {
        if(root == null) {
            return new BSTNode(val);
        }
        if(root.val == val) {
            root.dup++;
        } else if(root.val < val) {
            root.bigger++;
            root.right = addNode(root.right, val);
        } else {
            // root.val > val
            root.smaller++;
            root.left = addNode(root.left, val);
        }
        return root;
    }

    public int findKthNumber(int n, int k) {
        int i = 1;
        int num = 1, factor = 1;
        Stack<Integer> stack = new Stack<>();
        while(i < k && num * 10 <= n) {
            stack.push(num);
            num = num * 10;
            i++;
        }

        while(i < k) {
            if(num + factor * 9 < n && i + factor * 10 <= k) {
                while (stack.peek() + 1 > n) {
                    stack.pop();
                }
                num = stack.pop() + 1;
                i += factor * 10;
                factor = factor * 10;
            } else {
                if(factor != 1 && num * 10 <= n) {
                    stack.push(num);
                    num = num * 10;
                    factor /= 10;
                } else {
                    if(num + 1 <= n) {
                        num += 1;
                    } else {
                        num = stack.pop() + 1;
                        factor = factor * 10;
                    }
                }
                i++;
            }
        }
        return num;
    }

    public String nearestPalindromic(String n) {
        char[] chars = n.toCharArray();
        for(int i = chars.length / 2 + 1; i < chars.length; i++) {
            chars[i] = chars[chars.length - i];
        }
        return new String(chars);
    }

    private static final int M = 1000000000 + 7;
    public int checkRecord(int n) {
        if(n == 1) return 3;

        int[] P = new int[n + 1], L = new int[n + 1], A = new int[n + 1];
        int[] noAP = new int[n + 1], noAL = new int[n + 1];
        P[1] = L[1] = A[1] = 1;
        noAP[1] = noAL[1] = 1;
        noAL[2] = 2; L[2] = 3;
        for(int i = 2; i <= n; i++) {
            P[i] = ((A[i - 1] + L[i - 1]) % M + P[i - 1]) % M;
            if(i >= 3) L[i] = ((A[i - 1] + P[i - 1]) % M + (P[i - 2] + A[i - 2]) % M) % M;
            A[i] = (noAP[i - 1] + noAL[i - 1]) % M;
            noAP[i] = (noAL[i - 1] + noAP[i - 1] ) % M;
            if(i >= 3) noAL[i] = (noAP[i - 1] + noAP[i - 2]) % M;
        }
        return ((P[n] + L[n]) % M + A[n]) % M;
    }

    public int kInversePairs(int n, int k) {
        int[][] dp = new int[n + 1][k + 1];
        dp[1][0] = 1;
        for(int i = 2; i <= n; i++) {
            int sum = 0;
            for(int j = 0; j <= Math.min(k, i * (i - 1) / 2); j++) {
                if(j <= i - 1) {
                    sum = (sum + dp[i - 1][j]) % M;
                    dp[i][j] = sum;
                } else {
                    sum = (((sum - dp[i - 1][j - i]) % M + dp[i - 1][j]) % M + M) % M;
                    dp[i][j] = sum;
                }
            }
        }
        return dp[n][k];
    }

    public int scheduleCourse(int[][] courses) {
        int time = 0;
        Arrays.sort(courses, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }
        });
        Queue<Integer> pq = new PriorityQueue<>((a, b) -> (b - a));
        for(int[] course : courses) {
            time += course[0];
            pq.add(course[0]);
            if(time > course[1]) time -= pq.poll();
        }
        return pq.size();
    }

    public int findIntegers(int num) {
        String str = Integer.toBinaryString(num);
        int len = str.length();
        int[] a = new int[len], b = new int[len];
        a[0] = b[0] = 1; // len 2 -> idx = 1
        for(int i = 1; i < len; i++) {
            a[i] = a[i - 1] + b[i - 1];
            b[i] = a[i - 1];
        }
        int res = a[len - 1] + b[len - 1];
        for(int i = 0; i + 1 < len; i++) {
            if(str.charAt(i) == '1' && str.charAt(i + 1) == '1') {
                break;
            } else if(str.charAt(i) == '0' && str.charAt(i + 1) == '0') {
                // index i, force out i + 1 pos, substring len = len - i - 1, idx = len - i - 2
                res -= b[len - i - 2];
            }
        }
        return res;
    }

    public int strangePrinter(String s) {
        if(s == null || s.length() == 0) return 0;
        int len = s.length();
        int[][] dp = new int[len][len];
        for(int l = 1; l <= len; l++) {
            for(int i = 0, j = i + l - 1; i + l - 1 < len; i++, j++) {
                int res = l;
                // partition into s[i:k-1], s[k,j]
                for(int k = i + 1; k <= j; k++) {
                    int tmp = dp[i][k - 1] + dp[k][j];
                    if(s.charAt(i) == s.charAt(k)) tmp--;
                    res = Math.min(tmp, res);
                }
                dp[i][j] = res;
            }
        }
        return dp[0][len - 1];
    }

    public int[] findRedundantDirectedConnection(int[][] edges) {
        int n = edges.length;
        // node i -> idx i - 1
        int[] parent = new int[n], in = new int[n], direct = new int[n];
        List<Integer>[] child = new List[n];
        for(int i = 0; i < n; i++)  {
            parent[i] = i + 1;
            direct[i] = i + 1;
        }
        int[] c1 = null, c2 = null;
        for(int i = 0; i < edges.length; i++) {
            int[] edge = edges[i];
            if(find(edge[0], parent) != find(edge[1], parent) && in[edge[1] - 1] == 0) {
                parent[find(edge[1], parent) - 1] = find(edge[0], parent);
                direct[edge[1] - 1] = edge[0];
                in[edge[1] - 1]++;
            } else {
                if(in[edge[1] - 1] == 1) {
                    c1 = edge;
                    c2 = new int[]{direct[edge[1] - 1],edge[1]};
                } else {
                    //cycle detected
                    if(c1 != null) {
                        return c2;
                    }
                    Set<String> set = new HashSet<>();
                    set.add(edge[0] + " " + edge[1]);
                    int node = edge[0];
                    while(direct[node - 1] != node) {
                        if(set.add(direct[node - 1] + " " + node)) {
                            node = direct[node - 1];
                        } else {
                            break;
                        }
                    }
                    for(int j = i; j >= 0; j--) {
                        int[] res = edges[j];
                        if(set.contains(res[0] + " " + res[1])) return res;
                    }
                }
            }
        }
        return c1;
    }

    // give node, return the parent
    private int find(int node, int[] parent) {
        if(parent[node - 1] != node) {
            return find(parent[node - 1], parent);
        } else {
            return node;
        }
    }

    public int subarraySum(int[] nums, int k) {
        int sum = 0, res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for(int num : nums) {
            sum += num;
            res += map.getOrDefault(sum - k, 0);
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }

    public int search(int[] nums, int target) {
        int from = 0, to = nums.length - 1;
        while (from < to) {
            int mid = (from + to) / 2;
            if(target == nums[from]) return from;
            if(target == nums[to]) return to;
            if(nums[mid] == target) {
                return mid;
            } else if(nums[mid] < target){
                if(target > nums[from]) {
                    if(nums[mid] > nums[from]) {
                        from = mid + 1;
                    } else {
                        to = mid - 1;
                    }
                } else {
                    from = mid + 1;
                }
            } else {
                if(target > nums[from]) {
                    to = mid - 1;
                } else {
                    if(nums[mid] > nums[from]) {
                        from = mid + 1;
                    } else {
                        to = mid - 1;
                    }
                }
            }
        }
        return -1;
    }

    class UndirectedGraphNode {
        int label;
        List<UndirectedGraphNode> neighbors;
        UndirectedGraphNode(int x) { label = x; neighbors = new ArrayList<UndirectedGraphNode>(); }
    }

    public List<Integer> cheapestJump(int[] A, int B) {
        int[] cost = new int[A.length];
        int[] from = new int[A.length];
        Arrays.fill(cost, Integer.MAX_VALUE);
        cost[0] = 0;
        for(int i = 0; i < A.length; i++) {
            if(cost[i] == Integer.MAX_VALUE) continue;
            for(int jump = 1; jump <= B && i + jump < A.length; jump++) {
                int j = i + jump;
                if(A[j] != -1 && cost[i] + A[i] <= cost[j]) {
                    cost[j] = cost[i] + A[i];
                    if(from[j] != 0) {
                        from[j] = String.valueOf(i).compareTo(from[j] + "") < 0 ?
                            i : from[j];
                    } else {
                        from[j] = i;
                    }

                }
            }
        }
        if(cost[A.length - 1] == Integer.MAX_VALUE) {
            return new ArrayList<>();
        } else {
            List<Integer> res = new ArrayList<>();
            int pos = A.length - 1;
            while(pos != 0) {
                res.add(pos + 1);
                pos = from[pos];
            }
            res.add(1);
            Collections.reverse(res);
            return res;
        }
    }

    public int smallestDistancePair(int[] nums, int k) {
        int len = nums.length;
        Arrays.sort(nums);
        int lo = nums[1] - nums[0], hi = nums[len - 1] - nums[0];
        for(int i = 1; i + 1< len; i++) lo = Math.min(lo, nums[i + 1] - nums[i]);
        // find first appearence of the num that there are k part
        while(lo < hi) {
            int mid = (lo + hi) / 2;
            int sub = find(nums, mid);
            if(sub < k) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    private int find(int[] nums, int dis){
        int res = 0;
        for(int i = 0; i < nums.length; i++) {
            int j = find(nums, i, dis);
            res += j - i - 1;
        }
        return res;
    }

    private int find(int[] nums, int i, int dis) {
        // find first appearence num[res] - num[from] > dis
        int lo = i + 1, hi = nums.length;
        while(lo < hi) {
            int mid = (lo + hi) / 2;
            if(nums[mid] - nums[i] > dis) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    public void findArticulationPoint(int[][] edges, int V) {
        List<List<Integer>> adj = new ArrayList<>();
        for(int i = 0; i < V; i++) adj.add(new ArrayList<>());
        for(int[] edge : edges) {
            adj.get(edge[0]).add(edge[1]);
            adj.get(edge[1]).add(edge[0]);
        }
        int[] parent = new int[V], ancestor = new int[V], discover = new int[V];
        boolean[] visited = new boolean[V], res = new boolean[V];
        Arrays.fill(parent, -1);
        Arrays.fill(ancestor, Integer.MAX_VALUE);
        Arrays.fill(discover, Integer.MAX_VALUE);
        ancestor[0] = 0;
        findAP(parent, ancestor, discover, visited, res, 0, 0, adj);
        for(int i = 0; i < V; i++) {
            if(res[i]) System.out.println("Find AP: " + i);
        }
    }

    private void findAP(int[] parent, int[] ancestor, int[] discover, boolean[] visited,
                        boolean[] res, int step, int node, List<List<Integer>> adj) {
        ancestor[node] = discover[node] = step;
        visited[node] = true;
        int numOfChild = 0;
        for(int child : adj.get(node)) {
            if(visited[child]) {
                if(parent[node] != child) {
                    ancestor[node] = Math.min(ancestor[node], discover[child]);
                }
            } else {
                numOfChild++;
                parent[child] = node;
                findAP(parent, ancestor, discover, visited, res, step + 1, child, adj);
                ancestor[node] = Math.min(ancestor[child], ancestor[node]);
                if(parent[node] == -1 && numOfChild > 1) {
                    res[node] = true;
                }
                if(parent[node] != -1 && discover[node] <= ancestor[child]) {
                    res[node] = true;
                }
            }
        }
    }

    public int subArrayWithEqualZeroAndOne(int[] nums) {
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 0);
        int sum = 0, res = 0, from = -1, to = -1;
        for(int i = 1; i <= n; i++) {
            sum += nums[i - 1];
            int target = 2 * sum - i;
            if(map.containsKey(target)) {
                res = Math.max(res, i - map.get(target));
                from = map.get(target);
                to = i;
            } else {
                map.put(target, i);
            }
        }
        System.out.println(from + "  " + to);
        return res;
    }

    public List<Integer> numWithTwoOne(int n) {
        int cnt = 0;
        List<Integer> res = new ArrayList<>();
        for(int i = 1; i < 32; i++) {
            for(int j = 0; j < i; j++) {
                res.add((int)(Math.pow(2, i) + Math.pow(2, j)));
                if(++cnt == n) return res;
            }
        }
        return res;
    }

    public List<Integer> numWithMOne(int n, int m) {
        int end = m - 1;
        int size = 0;
        List<Integer> res = new ArrayList<>();
        List<Integer> prev = new ArrayList<>();
        while (size < n) {
            numWithMOne(res, prev, 0, m, end, n);
            size = res.size();
            end++;
        }
        return res;
    }

    private void numWithMOne(List<Integer> res, List<Integer> prev, int start, int remain, int end, int n) {
        if(remain == 1) {
            int sub = 0;
            for(int f : prev) {
                sub += (int)Math.pow(2, f);
            }
            sub += (int)Math.pow(2, end);
            res.add(sub);
            return;
        }
        for(int i = start; i <= end - remain + 1; i++) {
            prev.add(i);
            numWithMOne(res, prev, start + 1, remain - 1, end, n);
            prev.remove(prev.size() - 1);
            if(res.size() == n) {
                return;
            }
        }
    }

    private int numOfSubarrayDividedByK(int[] nums, int k) {
        int n = nums.length, sum = 0, res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for(int i = 1; i <= n; i++) {
            sum += nums[i - 1];
            int target = sum % k + k - k;
            res += map.getOrDefault(target, 0);
            map.put(sum % k, map.getOrDefault(sum % k, 0) + 1);
        }
        return res;
    }

    public int cutOffTree(List<List<Integer>> forest) {
        Queue<int[]> heights = new PriorityQueue<>((a, b) -> (a[2] - b[2]));
        int m = forest.size(), n = forest.get(0).size();
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                int height = forest.get(i).get(j);
                if(height > 0) {
                    heights.offer(new int[]{i, j,height});
                }
            }
        }
        int[] start = new int[3];
        int res = 0;
        while(!heights.isEmpty()) {
            int[] end = heights.poll();
            int step = bfs(forest, start, end, m, n);
            if(step == -1) {
                return -1;
            } else {
                res += step;
            }
            start = end;
        }
        return res;
    }

    private int bfs(List<List<Integer>> forest, int[] start, int[] end, int m, int n) {
        if(start[0] == end[0] && start[1] == end[1]) return 0;

        Queue<int[]> q = new LinkedList<>();
        boolean[][] visited = new boolean[m][n];
        q.add(start);
        visited[start[0]][start[1]] = true;
        int[][] dir = new int[][]{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        int step = 1;
        while (!q.isEmpty()) {
            int size = q.size();
            for(int loop = 0; loop < size; loop++) {
                int[] pos = q.poll();
                for(int i = 0; i < 4; i++) {
                    int x = pos[0] + dir[i][0], y = pos[1] + dir[i][1];
                    if(x >= 0 && x < forest.size() && y >= 0 && y < forest.get(x).size() &&
                            forest.get(x).get(y) != 0 && !visited[x][y]) {
                        if(x == end[0] && y == end[1]) {
                            return step;
                        } else {
                            q.add(new int[]{x, y});
                            visited[x][y] = true;
                        }
                    }
                }
            }
            step++;
        }
        return -1;
    }

    public int countPalindromicSubsequences(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for(int i = 0; i < n; i++) dp[i][i] = 1;
        for(int l = 2; l <= n; l++) {
            for(int i = 0; i + l - 1 < n; i++) {
                int j = i + l - 1;
                if(s.charAt(i) == s.charAt(j)) {
                    int lo = i + 1, hi = j - 1;
                    while(lo < j && s.charAt(lo) != s.charAt(j)) lo++;
                    while(hi > i && s.charAt(hi) != s.charAt(i)) hi--;
                    if(lo > hi) {
                        dp[i][j] = (dp[i + 1][j - 1] * 2) % M + 2;
                    } else if(lo == hi) {
                        dp[i][j] = (dp[i + 1][j - 1] * 2) % M + 1;
                    } else {
                        dp[i][j] = ((dp[i + 1][j - 1] * 2) % M - dp[lo + 1][hi - 1] + M) % M;
                    }
                } else {
                    dp[i][j] = ((dp[i + 1][j] + dp[i][j - 1]) % M + M - dp[i + 1][j - 1]) % M;
                }
            }
        }
        return dp[0][n - 1];
    }

    class Trie {
        boolean end;
        Trie[] children;
        public Trie() {
            children = new Trie[26];
        }
    }
    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        Trie root = new Trie();
        Arrays.sort(words, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return o1.length() - o2.length();
            }
        });
        List<String> res = new ArrayList<>();
        for(int i = 0; i < words.length; i++) {
            String word = words[i];
            if(check(root, word)) {
                res.add(word);
            } else {
                addWord(root, word);
            }
        }
        return res;
    }

    private void addWord(Trie root, String word) {
        Trie p = root;
        for(int i = 0; i < word.length(); i++) {
            if(p.children[word.charAt(i) - 'a'] == null) {
                p.children[word.charAt(i) - 'a'] = new Trie();
            }
            p = p.children[word.charAt(i) - 'a'];
        }
        p.end = true;
    }

    private boolean check(Trie root, String word) {
        Trie p = root;
        for(int i = 0; i < word.length(); i++) {
            if(p.children[word.charAt(i) - 'a'] == null) {
                return false;
            } else {
                p = p.children[word.charAt(i) - 'a'];
                if(p.end) {
                    if(check(root, word.substring(i + 1))) {
                        return true;
                    }
                }
            }
        }
        return p.end;
    }

    public String toHex(int num) {
        boolean reverse = num < 0;
        int[] bits = new int[32];
        int i = 0;
        num = Math.abs(num);
        while(num != 0) {
            bits[31 - i] = (num % 2 == 1 ? 1 : 0);
            i++;
            num = num / 2;
        }
        if(reverse) {
            for(int j = 0; j < 32; j++) bits[j] = bits[j] ^ 1;
            i = 31;
            while(bits[i] == 1) {
                bits[i] = 0;
                i--;
            }
            bits[i] = 1;
        }
        StringBuilder res = new StringBuilder();
        for(int j = 0; j < 32; j = j + 4) {
            int sub =  8 * bits[j] +  4 * bits[j + 1]
                    +  2 * bits[j + 2] +  bits[j + 3];
            char c = sub < 10 ? (char)('0' + sub) : (char)('a' + sub - 10);
            if(res.length() > 0 || c != '0') res.append(c);
        }
        return res.toString();
    }

    public boolean isAdditiveNumber(String num) {
        for(int i = 1; i < num.length(); i++) {
            for(int j = i + 1; Math.max(i, j - i) <= num.length() - j; j++) {
                if(valid(i, j, num)) return true;
            }
        }
        return false;
    }

    private boolean valid(int i , int j, String num) {
        if(num.charAt(0) == '0' && i > 1) return false;
        if(num.charAt(i) == '0' && j > i + 1) return false;
        String sum, num1 = num.substring(0, i), num2 = num.substring(i, j);
        for(int k = j; k < num.length(); k = k + sum.length()) {
            sum = stringAdd(num1, num2);
            num1 = num2;
            num2 = sum;
            if(!num.substring(k).startsWith(sum)) return false;
        }
        return true;
    }

    private String stringAdd(String num1, String num2) {
        int i = 0, carry = 0;
        StringBuilder sb = new StringBuilder();
        while(i < num1.length() || i < num2.length()) {
            int digit1 = num1.length() - i - 1 >= 0 ? num1.charAt(num1.length() - i - 1) - '0' : 0;
            int digit2= num2.length() - i - 1 >= 0 ? num2.charAt(num2.length() - i - 1) - '0' : 0;
            sb.append((digit1 + digit2 + carry) % 10);
            carry = (digit1 + digit2 + carry) / 10;
            i++;
        }
        if(carry == 1) sb.append(1);
        return sb.reverse().toString();
    }

    public String splitLoopedString(String[] strs) {
        StringBuilder sb = new StringBuilder();
        for(String str : strs) {
            String reverse = new StringBuilder(str).reverse().toString();
            if(reverse.compareTo(str) > 0) {
                sb.append(reverse);
            } else {
                sb.append(str);
            }
        }
        int idx = 0;
        String ori = sb.toString();
        String max = ori;
        for(int i = 0; i < strs.length; i++) {
            String str = strs[i];
            String middle = ori.substring(idx + str.length()) + ori.substring(0, idx);
            for(int j = 0; j <= str.length(); j++) {
                String s = str.substring(j) + middle + str.substring(0, j);
                if(s.compareTo(max) > 0) max = s;
            }
            String reverse = new StringBuilder(str).reverse().toString();
            for(int j = 0; j <= str.length(); j++) {
                String s = reverse.substring(j) + middle + reverse.substring(0, j);
                if(s.compareTo(max) > 0) max = s;
            }
            idx += str.length();
        }
        return max;
    }

    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = 0, max = Integer.MIN_VALUE;
        Arrays.sort(nums);
        for(int num : nums) {
            sum += num;
            max = max > num ? max : num;
        }
        if(sum % k != 0 || max > sum / k) return false;

        boolean[] visited = new boolean[nums.length];
        boolean res = partition(nums, visited, k, 0, sum / k);
        return res;
    }

    private boolean partition(int[] nums, boolean[] visited, int remain, int preSum, int partitionSum) {
        if(remain == 0) {
            return true;
        }
        for(int i = nums.length - 1; i >= 0; i--) {
            if(!visited[i]) {
                if(preSum + nums[i] < partitionSum) {
                    visited[i] = true;
                    if(partition(nums, visited, remain, preSum + nums[i], partitionSum)) {
                        return true;
                    } else {
                        visited[i] = false;
                    }
                } else if(preSum + nums[i] == partitionSum) {
                    visited[i] = true;
                    if(partition(nums, visited, remain - 1, 0, partitionSum)) {
                        return true;
                    } else {
                        visited[i] = false;
                    }
                }
            }
        }
        return false;
    }

    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if(n == 1) return Collections.singletonList(0);
        List<Set<Integer>> adj = new ArrayList<>();
        for(int i = 0; i < n; i++) {
            adj.add(new HashSet<>());
        }
        for(int[] edge : edges) {
            adj.get(edge[0]).add(edge[1]);
            adj.get(edge[1]).add(edge[0]);
        }
        List<Integer> leaves = new ArrayList<>();
        for(int i = 0; i < n; i++) {
            if(adj.get(i).size() == 1) leaves.add(i);
        }
        while(n > 2) {
            n = n - leaves.size();
            List<Integer> next = new ArrayList<>();
            for(int leaf : leaves) {
                int nei = adj.get(leaf).iterator().next();
                adj.get(nei).remove(leaf);
                if(adj.get(nei).size() == 1) {
                    next.add(nei);
                }
            }
            leaves = next;
        }
        return leaves;
    }

    public char[][] updateBoard(char[][] board, int[] click) {
        if(board[click[0]][click[1]] == 'M') {
            board[click[0]][click[1]] = 'X';
            return board;
        } else {
            boolean[][] visited = new boolean[board.length][board[0].length];
            bfs(board, click, visited);
            return board;
        }
    }

    private void bfs(char[][] board, int[] click, boolean[][] visited) {
        int[] xx = new int[]{-1, 0, 1}, yy = new int[]{-1, 0, 1};
        int mines = 0;
        visited[click[0]][click[1]] = true;
        for(int i = 0; i < xx.length; i++) {
            for(int j = 0; j < yy.length; j++) {
                int x = click[0] + xx[i], y = click[1] + yy[j];
                if(x >= 0 && x < board.length && y >= 0 && y < board[0].length && board[x][y] == 'M') {
                    mines++;
                }
            }
        }
        if(mines == 0) {
            board[click[0]][click[1]] = 'B';
            for(int i = 0; i < xx.length; i++) {
                for(int j = 0; j < yy.length; j++) {
                    int x = click[0] + xx[i], y = click[1] + yy[j];
                    if(x >= 0 && x < board.length && y >= 0 && y < board[0].length && !visited[x][y]) {
                        bfs(board, new int[]{x, y}, visited);
                    }

                }
            }
        } else {
            board[click[0]][click[1]] = (char)(mines + '0');
        }
    }

    public List<Integer> largestDivisibleSubset(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if(nums == null || nums.length == 0) return res;
        int[] parent = new int[nums.length];
        int[] size = new int[nums.length];
        Arrays.sort(nums);
        Arrays.fill(parent, - 1);
        int maxSize = 1, maxIdx = 0;
        for(int i = 0; i < nums.length; i++) {
            size[i] = 1;
            for(int j = 0; j < i; j++) {
                if(nums[i] % nums[j] == 0) {
                    if(size[j] + 1 > size[i]) {
                        size[i] = size[j] + 1;
                        parent[i] = j;
                    }
                    if(size[i] > maxSize) {
                        maxSize = size[i];
                        maxIdx = i;
                    }
                }
            }
        }
        int i = maxIdx;
        while(i != -1) {
            res.add(nums[i]);
            i = parent[i];
        }
        return res;
    }

    public boolean makesquare(int[] nums) {
        int sum = 0, max = 0;
        for(int num : nums)  {
            sum += num;
            max = Math.max(max, num);
        }
        if(sum % 4 != 0 || max > sum / 4) return false;
        Arrays.sort(nums);
        boolean[] visited = new boolean[nums.length];
        return makeSquare(nums, visited, 0, sum / 4, 4);
    }

    private boolean makeSquare(int[] nums, boolean[] visited, int preSum, int target, int remain) {
        if(remain == 0) {
            return true;
        }
        for(int i = nums.length - 1; i >= 0; i--) {
            if(!visited[i]) {
                if(preSum + nums[i] == target) {
                    visited[i] = true;
                    if(makeSquare(nums, visited, 0, target, remain - 1)){
                        return true;
                    } else {
                        visited[i] = false;
                    }
                } else if(preSum + nums[i] < target) {
                    visited[i] = true;
                    if(makeSquare(nums, visited, preSum + nums[i], target, remain)){
                        return true;
                    } else {
                        if(preSum == 0) {
                            return false;
                        } else {
                            visited[i] = false;
                        }
                    }
                }
            }
        }
        return false;
    }

    public int longestLine(int[][] M) {
        if(M == null || M.length == 0 || M[0].length == 0) return 0;
        int row = M.length, col = M[0].length;
        int[][] verti = new int[row + 1][col + 1], hori = new int[row + 1][col + 1],
                diag = new int[row + 1][col + 1], anti = new int[row + 1][col + 1];
        int max = 0;
        for(int i = 1; i <= row; i++) {
            for(int j = 1; j <= col; j++) {
                if(M[i - 1][j - 1] == 1) {
                    verti[i][j] = verti[i - 1][j] + 1;
                    hori[i][j]  = hori[i][j - 1] + 1;
                    diag[i][j]  = diag[i - 1][j - 1] + 1;
                    anti[i][j] = anti[i - 1][j + 1] + 1;
                    int sub1 = Math.max(verti[i][j], hori[i][j]);
                    int sub2 = Math.max(diag[i][j], anti[i][j]);
                    max = Math.max(Math.max(sub1, sub2), max);
                }
            }
        }
        return max;
    }

    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for(int i = 0; i < coins.length; i++) {
            for(int total = 1; total <= amount; total++) {
                if(coins[i] <= total) {
                    dp[total] += dp[total - coins[i]];
                }
            }
        }
        return dp[amount];
    }

    class Employee {
        // It's the unique id of each node;
        // unique id of this employee
        public int id;
        // the importance value of this employee
        public int importance;
        // the id of direct subordinates
        public List<Integer> subordinates;
    }

    public int getImportance(List<Employee> employees, int id) {
        int sum = 0;
        Map<Integer, Employee> map = new HashMap<>();
        for(Employee e : employees) {
            map.put(e.id, e);
        }
        Queue<Integer> q = new LinkedList<>();
        Set<Integer> visited = new HashSet<>();
        q.add(id);
        visited.add(id);
        while (!q.isEmpty()) {
            Employee e = map.get(q.poll());
            sum += e.importance;
            for(Integer sub :e.subordinates) {
                if(visited.add(sub)) {
                    q.add(sub);
                }
            }
        }
        return sum;
    }

    public boolean isOneBitCharacter(int[] bits) {
        return valid(bits, bits.length - 2);
    }

    private boolean valid(int[] bits, int idx) {
        if(idx == 0) {
            if(bits[idx] == 0) {
                return true;
            } else {
                return false;
            }
        } else if(idx == 1) {
            if (bits[idx] == 0) {
                return true;
            } else {
                if (bits[idx - 1] == 1) {
                    return true;
                } else {
                    return false;
                }
            }
        }

        if(bits[idx] == 0) {
            if(bits[idx - 1] == 1 && valid(bits, idx - 2)) {
                return true;
            }
            return valid(bits, idx - 1);
        } else {
            //bits[i] == 1
            if(bits[idx - 1] == 0) {
                return false;
            } else {
                return valid(bits, idx - 2);
            }
        }
    }

    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        int lo = 0, hi = 0, d = 0, len = 0;
        int[] count = new int[256];
        while(hi < s.length()) {
            if(count[s.charAt(hi++)]++ == 0) d++;
            len = Math.max(len, hi - lo);
            while(d > k) {
                if(--count[s.charAt(lo++)] == 0){
                    d--;
                }
            }
        }
        return len;
    }

    public String minWindow(String s, String t) {
        int start = 0, minLen = Integer.MAX_VALUE, lo = 0, hi = 0, remain = t.length();
        int[] count = new int[256];
        for(char c : t.toCharArray()) {
            count[c]++;
        }
        while(hi < s.length()) {
            //if(counter[s.charAt(hi++)]-- > 0) remain--;
            if(count[s.charAt(hi++)]-- > 0) remain--;
            while(remain == 0) {
                if(hi - lo < minLen) {
                    minLen = hi - lo;
                    start = lo;
                }
                if(++count[s.charAt(lo++)] > 0) remain++;
            }
        }

        return minLen == Integer.MAX_VALUE ? "" : s.substring(start, start + minLen);
    }

    public int characterReplacement(String s, int k) {
        int lo = 0, hi = 0, maxCount = 0, maxlen = 0;
        int[] count = new int[26];
        while(hi < s.length()) {
            maxCount = Math.max(maxCount, ++count[s.charAt(hi++) - 'A']);
            while(hi - lo - maxCount > k) {
                count[s.charAt(lo++) - 'A']--;
            }
            maxlen = Math.max(maxlen, hi - lo);
        }
        return maxlen;
    }

    public String toGoatLatin(String S) {
        String[] words = S.split(" ");
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < words.length; i++) {
            sb.append(convertWord(words[i], i + 1));
            if(i < words.length - 1) sb.append(" ");
        }
        return sb.toString();
    }

    private StringBuilder convertWord(String word, int idx) {
        char c = word.charAt(0);
        StringBuilder sb = new StringBuilder();
        if(c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'A'
                || c == 'E' || c == 'I' || c == 'O' || c == 'U') {
            sb.append(word).append("ma");
        } else {
            sb.append(word.substring(1)).append(c).append("ma");
        }
        while (idx-- > 0) {
            sb.append('a');
        }
        return sb;
    }

    public int numFriendRequests(int[] ages) {
        Arrays.sort(ages);
        int res = 0;
        int last = -1, lastIdx = -1;
        for(int i = ages.length - 1; i >= 0; i--) {
            double target = (0.5 * ages[i]) + 7;
            int idx = binarySearch(ages, target, i);
            res += i - idx;
            if(ages[i] == last && ages[i] > 14) {
                res += lastIdx - i;
            } else {
                last = ages[i];
                lastIdx = i;
            }

        }
        return res;
    }

    class Task{
        int difficulty;
        int profit;
        public Task(int difficulty, int profit) {
            this.difficulty = difficulty;
            this.profit = profit;
        }
    }

    public int maxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
        int n = profit.length;
        Task[] tasks = new Task[n];
        for(int i = 0; i < n; i++) {
            tasks[i] = new Task(difficulty[i], profit[i]);
        }
        Arrays.sort(tasks, new Comparator<Task>() {
            @Override
            public int compare(Task o1, Task o2) {
                return o1.difficulty - o2.difficulty;
            }
        });
        int maxProfit = tasks[0].profit;
        for(int i = 1; i < n; i++) {
            if(maxProfit < tasks[i].profit) maxProfit = tasks[i].profit;
            tasks[i].profit = maxProfit;
        }
        int total = 0;
        for(int i = 0; i < worker.length; i++) {
            int firstCannot = binarySearch(tasks, worker[i], n);
            if(firstCannot != 0) total += tasks[firstCannot - 1].profit;
        }
        return total;
    }

    private int binarySearch(Task[] task, int difficulty, int n) {
        if(task[n - 1].difficulty <= difficulty) {
            return n;
        }
        int lo = 0, hi = n - 1;
        while(lo < hi) {
            int mid = (lo + hi) / 2;
            if(task[mid].difficulty <= difficulty) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    private int binarySearch(int[] ages, double target, int to) {
        if(ages[0] > target) return 0;
        int lo = 0, hi = to;
        while(lo < hi) {
            int mid = (lo + hi) / 2;
            if(ages[mid] <= target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    public int largestIsland(int[][] grid) {
        int row = grid.length, col = grid[0].length;
        boolean zero = false;
        Map<String, Integer> map = new HashMap<>();
        Map<String, Integer> graphIdx = new HashMap<>();
        int idx = 1, max = 0;
        for(int i = 0; i < row; i++) {
            for(int j = 0; j < col; j++) {
                if(grid[i][j] == 1 && !map.containsKey(i + " " + j)) {
                    Set<String> visited = new HashSet<>();
                    bfs(grid, row, col, i, j, visited);
                    for(String pos : visited) {
                        map.put(pos, visited.size());
                        graphIdx.put(pos, idx);
                    }
                    idx++;
                }
            }
        }
        for(int i = 0; i < row; i++) {
            for(int j = 0; j < col; j++) {
                if(grid[i][j] == 0) {
                    zero = true;
                    int size = size(map, graphIdx, i, j,row, col) + 1;
                    max = Math.max(size, max);
                }
            }
        }
        if(!zero) {
            return row * col;
        } else {
            return max;
        }
    }

    private int size(Map<String, Integer> map, Map<String, Integer> graphIdx, int i, int j, int row, int col) {
        int res = 0;
        Set<Integer> set = new HashSet<>();
        if(i - 1 >= 0 && map.containsKey((i - 1) + " " + j) && !set.contains(graphIdx.get((i - 1) + " " + j))) {
            set.add(graphIdx.get((i - 1) + " " + j));
            res += map.get((i - 1) + " " + j);
        }
        if(i + 1 < row && map.containsKey((i + 1) + " " + j) && !set.contains(graphIdx.get((i + 1) + " " + j))) {
            set.add(graphIdx.get((i + 1) + " " + j));
            res += map.get((i + 1) + " " + j);
        }
        if(j - 1 >= 0 && map.containsKey(i + " " + (j - 1)) && !set.contains(graphIdx.get(i + " " + (j - 1)))) {
            set.add(graphIdx.get(i + " " + (j - 1)));
            res += map.get(i + " " + (j - 1));
        }
        if(j + 1 < col && map.containsKey(i + " " + (j + 1)) && !set.contains(graphIdx.get(i + " " + (j + 1)))) {
            res += map.get(i + " " + (j + 1));
        }
        return res;
    }

    private void bfs(int[][] grid, int row, int col, int x, int y, Set<String> visited) {
        int[][] dir = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        Queue<int[]> q = new LinkedList<>();
        q.add(new int[]{x, y});
        visited.add(x + " " + y);
        while (!q.isEmpty()) {
            int[] pos = q.poll();
            for(int i = 0; i < 4; i++) {
                int newX = pos[0] + dir[i][0], newY = pos[1] + dir[i][1];
                if(newX >= 0 && newX < row && newY >= 0 && newY < col
                        && grid[newX][newY] == 1 && visited.add(newX + " " + newY)) {
                    q.add(new int[]{newX, newY});
                }
            }
        }
    }

    public int calculateI(String s) {
        int res = 0, oper = 1, val = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(1);
        for(int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if(c >= '0' && c <= '9') {
                val = val * 10 + c - '0';
            } else if(c == '-' || c == '+') {
                res += val * oper;
                oper = (c == '+' ? 1 : -1) * stack.peek();
                val = 0;
            } else if(c == '(') {
                stack.push(oper);
            } else if(c == ')') {
                stack.pop();
            }
        }
        res += val * oper;
        return res;
    }

    public int calculate(String s) {
        int n1 = 0, n2 = 1, o1 = 1, o2 = 1, i = 0, val = 0;
        while(i < s.length()) {
            char c = s.charAt(i);
            if(c >= '0' && c <= '9') {
                while(i < s.length() && Character.isDigit(s.charAt(i))) {
                    val = val * 10 + s.charAt(i) - '0';
                    i++;
                }
                n2 = o2 == 1 ? n2 * val : n2 / val;
                val = 0;
            } else if(c == '-' || c == '+') {
                // use the last secondary oper
                n1 = n1 + o1 * n2;
                o1 = c == '+' ? 1 : -1;
                n2 = 1;
                o2 = 1;
                i++;
            } else if(c == '*' || c == '/') {
                // use the last firstLevel oper
                o2 = c == '*' ? 1 : -1;
                i++;
            } else {
                i++;
            }
        }
        n1 = n1 + o1 * n2;
        return n1;
    }

    public List<String> topKFrequent(String[] words, int k) {
        List<String> list = new ArrayList<>();
        if(words == null || words.length == 0 || k == 0) return list;
        Map<String, Integer> map = new HashMap<>();
        for(String word : words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        Comparator<Map.Entry<String, Integer>> comparator = new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                return o1.getValue() == o2.getValue() ? o2.getKey().compareTo(o1.getKey()) : o1.getValue() - o2.getValue();
            }
        };
        Queue<Map.Entry<String, Integer>> pq = new PriorityQueue<>(k, comparator);
        for(Map.Entry<String, Integer> entry : map.entrySet()) {
            if(pq.size() < k) {
                pq.add(entry);
            } else {
                Map.Entry<String, Integer> peek = pq.peek();
                if(entry.getValue() > peek.getValue() ||
                        (entry.getValue() == peek.getValue() && entry.getKey().compareTo(peek.getKey()) < 0)) {
                    pq.poll();
                    pq.add(entry);
                }
            }

        }
        while (!pq.isEmpty()) {
            list.add(pq.poll().getKey());
        }
        Collections.reverse(list);
        return list;
    }

    public List<int[]> getSkyline(int[][] buildings) {
        List<int[]> res = new ArrayList<>();
        if(buildings == null || buildings.length == 0) return res;

        List<int[]> pos = new ArrayList<>();
        for(int[] building : buildings) {
            pos.add(new int[]{building[0], building[2]});
            pos.add(new int[]{building[1], -building[2]});
        }
        Collections.sort(pos, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0];
            }
        });

        Queue<Integer> pq = new PriorityQueue<>((a, b) -> (b - a));
        pq.add(0);
        int prev = 0;
        for(int[] p : pos) {
            if(p[1] < 0) {
                // reach the end point of building
                pq.remove(-p[1]);
                if(pq.peek() != prev) {
                    res.add(new int[]{p[0], pq.peek()});
                    prev = pq.peek();
                }
            } else {
                // reach the start point of building
                pq.add(p[1]);
                if(pq.peek() != prev) {
                    res.add(new int[]{p[0], pq.peek()});
                    prev = pq.peek();
                }
            }
        }
        return res;
    }

    class Segment{
        int start, end, height;

        Segment(int s, int e, int h) {
            start = s;
            end = e;
            height = h;
        }
    }

    public List<Integer> fallingSquaresBrute(int[][] positions) {

        List<Integer> res = new ArrayList<>();
        TreeSet<Segment> bst = new TreeSet<>((s1, s2) -> Integer.compare(s1.start, s2.start));
        int max = 0;

        for(int[] p : positions) {
            Segment current = new Segment(p[0], p[0] + p[1], p[1]);
            Segment left = bst.lower(current);
            if(left != null && left.end > current.start) {
                bst.add(new Segment(current.start, left.end, left.height));
                left.end = current.start;
            }

            Segment right = bst.lower(new Segment(current.end, 0, 0));
            if(right != null && right.end > current.end)
                bst.add(new Segment(current.end, right.end, right.height));

            Set<Segment> sub = bst.subSet(current, true, new Segment(current.end, 0, 0), false);
            for(Iterator<Segment> i = sub.iterator(); i.hasNext(); i.remove())
                current.height = Math.max(current.height, i.next().height + p[1]);

            bst.add(current);

            res.add(max = Math.max(max, current.height));
        }

        return res;
    }

    public List<Integer> fallingSquares(int[][] positions) {
        List<Integer> res = new ArrayList<>();
        TreeMap<Integer, Integer> startHeight = new TreeMap<>();
        startHeight.put(0, 0);
        int max = 0;
        for (int[] pos : positions) {
            int start = pos[0], end = start + pos[1];
            Integer from = startHeight.floorKey(start);
            int height = startHeight.subMap(from, end).values().stream().max(Integer::compare).get() + pos[1];
            max = Math.max(height, max);
            res.add(max);
            // remove interval within [start, end)
            int lastHeight = startHeight.floorEntry(end).getValue();
            startHeight.put(start, height);
            startHeight.put(end, lastHeight);
            startHeight.keySet().removeAll(new HashSet<>(startHeight.subMap(start, false, end, false).keySet()));
        }
        return res;
    }

    class SegmentTree {
        int N, H;
        int[] tree, lazy;

        SegmentTree(int N) {
            this.N = N;
            H = 1;
            while ((1 << H) < N) H++;
            tree = new int[2 * N];
            lazy = new int[N];
        }

        private void apply(int x, int val) {
            tree[x] = Math.max(tree[x], val);
            if (x < N) lazy[x] = Math.max(lazy[x], val);
        }

        private void pull(int x) {
            while (x > 1) {
                x >>= 1;
                tree[x] = Math.max(tree[x * 2], tree[x * 2 + 1]);
                tree[x] = Math.max(tree[x], lazy[x]);
            }
        }

        private void push(int x) {
            for (int h = H; h > 0; h--) {
                int y = x >> h;
                if (lazy[y] > 0) {
                    apply(y * 2, lazy[y]);
                    apply(y * 2 + 1, lazy[y]);
                    lazy[y] = 0;
                }
            }
        }

        public void update(int L, int R, int h) {
            L += N; R += N;
            int L0 = L, R0 = R, ans = 0;
            while (L <= R) {
                if ((L & 1) == 1) apply(L++, h);
                if ((R & 1) == 0) apply(R--, h);
                L >>= 1; R >>= 1;
            }
            pull(L0); pull(R0);
        }

        public int query(int L, int R) {
            L += N; R += N;
            int ans = 0;
            push(L); push(R);
            while (L <= R) {
                if ((L & 1) == 1) ans = Math.max(ans, tree[L++]);
                if ((R & 1) == 0) ans = Math.max(ans, tree[R--]);
                L >>= 1; R >>= 1;
            }
            return ans;
        }
    }

    class BSTNode {
        long val;
        int smaller;
        int bigger;
        int dup;
        int h;
        BSTNode left = null, right = null;
        public BSTNode (long val){
            this.val = val;
            this.smaller = 0;
            this.bigger = 0;
            this.dup = 1;
            this.h = 0;
        }
    }

    class StringNode {
        String word;
        StringNode[] children;
        public StringNode(String word) {
            this.word = word;
            children = new StringNode[26];
        }
    }

    public String longestWord(String[] words) {
        if(words == null || words.length == 0) return "";
        Arrays.sort(words, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return o1.length() == o2.length() ? o1.compareTo(o2) : o1.length() - o2.length();
            }
        });
        if(words[0].length() > 1) return "";

        StringNode root = new StringNode("");
        String res = "";
        for(String word : words) {
            if(insert(root, word) && word.length() > res.length()) {
                res = word;
            }
        }
        return res;
    }

    private boolean insert(StringNode root, String word) {
        StringNode p = root;
        for(int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if(p.children[c - 'a'] != null) {
                p = p.children[c - 'a'];
            } else {
                if(i == word.length() - 1) {
                    p.children[c - 'a'] = new StringNode(word);
                    return true;
                } else {
                    return false;
                }
            }
        }
        return false;
    }

    public List<String> removeComments(String[] source) {
        List<String> res = new ArrayList<>();

        int row = 0, col = 0;
        boolean blocked = false;
        StringBuilder line = new StringBuilder();

        while(row < source.length) {
            if(blocked) {
                int idx = source[row].indexOf("*/", col);
                if(idx == -1) {
                    row++;
                    col = 0;
                } else {
                    blocked = false;
                    col = idx + 2;
                }
            } else {
                int idx1 = source[row].indexOf("/*", col);
                int idx2 = source[row].indexOf("//", col);

                if(idx1 == -1) idx1 = source[row].length();
                if(idx2 == -1) idx2 = source[row].length();

                for(int i = col; i < Math.min(idx1, idx2); i++) {
                    line.append(source[row].charAt(i));
                }

                if(idx2 <= idx1) {
                    row++;
                    col = 0;
                    if(line.length() > 0) {
                        res.add(line.toString());
                        line.setLength(0);
                    }
                } else {
                    blocked = true;
                    col = idx1 + 2;
                }
            }
        }
        return res;
    }

    public int[][] candyCrush(int[][] board) {
        Set<String> set = new HashSet<>();
        for(int i = 0; i < board.length; i++) {
            for(int j = 0; j < board[0].length; j++) {
                int candy = board[i][j];
                if(candy != 0) {
                    if((i - 2 >= 0 && board[i - 2][j] == candy && board[i - 1][j] == candy) ||
                            (i - 1 >= 0 && i + 1 < board.length && board[i - 1][j] == candy && board[i + 1][j] == candy) ||
                            (i + 2 < board.length && board[i + 1][j] == candy && board[i + 2][j] == candy) ||
                            (j - 2 >= 0 && board[i][j - 2] == candy && board[i][j - 1] == candy) ||
                            (j - 1 >= 0 && j + 1 < board[0].length && board[i][j - 1] == candy && board[i][j + 1] == candy) ||
                            (j + 2 < board[0].length && board[i][j + 1] == candy && board[i][j + 2] == candy)) {
                        set.add(i + " " + j);
                    }
                }
            }
        }
        if(set.size() == 0) {
            return board;
        }
        drop(board, set);
        return candyCrush(board);
    }

    private void drop(int[][] board, Set<String> set) {
        for(String pos : set) {
            String[] xy = pos.split(" ");
            int x = Integer.parseInt(xy[0]), y = Integer.parseInt(xy[1]);
            board[x][y] = 0;
        }
        for(int j = 0; j < board[0].length; j++) {
            int lo = board.length - 1, hi = board.length - 1;
            while (hi >= 0) {
                if(board[hi][j] == 0) {
                    hi--;
                } else {
                    board[lo--][j] = board[hi--][j];
                }
            }
            while(lo >= 0) board[lo--][j] = 0;
        }
    }

    public int overLap(int A, int B, int C, int D, int E, int F, int G, int H) {
        if(E >= C || G <= A || F >= D || H <= B) return 0;

        int blx = Math.max(A, E), trx = Math.min(G, C);
        int bly = Math.max(B, F), tryy = Math.min(D, H);

        return (trx - blx) * (tryy - bly);
    }

    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        return (C - A) * (D - B) + (G - E) * (H - F) - overLap(A, B, C, D, E, F, G , H);
    }

    public int countDigitOne(int n) {
        return 0;
    }

    public List<List<Integer>> largeGroupPositions(String s) {
        List<List<Integer>> res = new ArrayList<>();
        char prev = s.charAt(0);
        int len = 1, start = 0;
        for(int i = 1; i < s.length(); i++) {
            char c = s.charAt(i);
            if(c == prev) {
                len++;
            } else {
                int groupLen = i - start;
                if(groupLen >= 3) {
                    res.add(Arrays.asList(start, i - 1));
                }
                start = i;
                len = 1;
                prev = c;
            }
        }
        if(s.length() - start >= 3) res.add(Arrays.asList(start, s.length() - 1));
        return res;
    }

    public String maskPII(String s) {
        if(s == null || s.length() < 8) return "";

        char c = s.charAt(0);
        if((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            return email(s);
        } else {
            return phone(s);
        }
    }

    private String email(String str) {
        str = str.toLowerCase();
        int idx = str.indexOf('@');
        StringBuilder sb = new StringBuilder();
        sb.append(str.charAt(0)).append("*****").append(str.substring(idx - 1));
        return sb.toString();
    }

    private String phone(String str) {
        List<Character> list = new ArrayList<>();
        for(char c : str.toCharArray()) {
            if(c >= '0' && c <= '9') {
                list.add(c);
            }
        }
        StringBuilder sb = new StringBuilder();
        if(list.size() == 10) {
            sb.append("***-***-").append(list.get(6)).append(list.get(7)).append(list.get(8)).append(list.get(9));
        } else {
            sb.append('+');
            int i = list.size() - 10;
            while(i-- > 0) sb.append('*');
            int n = list.size();
            sb.append("-***-***-").append(list.get(n - 4)).append(list.get(n - 3)).append(list.get(n - 2)).append(list.get(n - 1));
        }
        return sb.toString();
    }

    public int consecutiveNumbersSum(int n) {
        int res = 0;
        for(int k = 1; k * k + k <= 2 * n; k++) {
            if(k % 2 == 1) {
                if(n % k == 0 && n / k >= (k + 1) / 2) res++;
            } else {
                if((n - k / 2) % k == 0 && ((n - k / 2) / k) >= k / 2) res++;
            }
        }
        return res;
    }

    public int uniqueLetterString(String s) {
        Map<Character, List<Integer>> map = new HashMap<>();
        for(int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if(!map.containsKey(c)) {
                map.put(c, new ArrayList<>());
            }
            map.get(c).add(i);
        }

        long res = 0;
        for(Character c : map.keySet()) {
            List<Integer> list = map.get(c);
            for(int i = 0; i < list.size(); i++) {
                long n1 = (long)list.get(i) - (i - 1 >= 0 ? list.get(i - 1) : -1);
                long n2 = (i + 1 < list.size() ? list.get(i + 1) : s.length()) - (long)list.get(i);
                res = (res + (n1 * n2)) % M;
            }
        }
        return (int)res;
    }

    public boolean backspaceCompare(String S, String T) {
        if(S == null && T == null) {
            return true;
        } else if(S == null || T == null) {
            return false;
        } else {
            return cleanString(S).equals(cleanString(T));
        }
    }

    private String cleanString(String str) {
        Stack<Character> stack = new Stack<>();

        for(char c : str.toCharArray()) {
            if(c == '#') {
                if(!stack.isEmpty()) {
                    stack.pop();
                }
            } else {
                stack.push(c);
            }
        }
        StringBuilder sb = new StringBuilder();
        while(!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        return sb.toString();
    }

    public int longestMountain(int[] A) {
        if (A == null || A.length < 3) return 0;
        int i = 0, maxLen = 0;
        while(i + 1 < A.length) {
            int lo = i;
            boolean up = false;
            while(i + 1 < A.length && A[i] < A[i + 1]) {
                up = true;
                i++;
            }
            if(i == A.length - 1 || !up || A[i] == A[i + 1]) {
                i++;
                continue;
            }

            while(i + 1 < A.length && A[i] > A[i + 1]) {
                i++;
            }
            if(maxLen < i - lo + 1) {
                maxLen = i - lo + 1;
            }
        }
        return maxLen;
    }

    public boolean isNStraightHand(int[] hand, int W) {
        if(hand.length % W != 0) return false;

        TreeMap<Integer, Integer> map = new TreeMap<>();
        for(int num : hand) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        int group = hand.length / W, i = 0;
        while(i++ < group) {
            int minkey = map.firstKey(), offset = 0;

            if(map.get(minkey) == 1) {
                map.remove(minkey);
            } else {
                map.put(minkey, map.get(minkey) - 1);
            }

            while(++offset < W) {
                int n = minkey + offset;
                if(map.containsKey(n)) {
                    int val = map.get(n);
                    if(val == 1) {
                        map.remove(n);
                    } else {
                        map.put(n, val - 1);
                    }
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {

        return null;
    }
    
    public static void main(String[] args) {
        Solution solution = new Solution();
        Random r = new Random();

        int ran = 0;
        for (int i = 0; i < 50; i++) {
            ran =r.nextInt(2272);
        }

        System.out.println(ran);
    }
}

