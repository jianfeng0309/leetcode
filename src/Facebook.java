import sun.jvm.hotspot.debugger.windbg.DLL;

import java.util.*;

/**
 * Created by GuoJianFeng on 11/2/17.
 */
public class Facebook {

    public static class ListNode {
        int val;
        ListNode next;
        ListNode down;
        ListNode(int x) { val = x; }
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    private static class DLLNode{
        int val;
        DLLNode left;
        DLLNode right;
        public DLLNode(int val) {
            this.val = val;
            this.left = null;
            this.right = null;
        }

        public DLLNode(DLLNode left, DLLNode right) {
            this.left = left;
            this.right = right;
        }
    }

    public int[] sortedSquares(int[] nums) {
        int n = nums.length;
        int[] sorted = new int[n];
        int i = 0, j = n - 1;
        int index = n-1;
        while (i <= j) {
            sorted[index--]= nums[i]*nums[i]>=nums[j]*nums[j]? nums[i]*nums[i++]:nums[j]*nums[j--];
        }
        return sorted;
    }

    // Given a string, find the minimum number of characters to be inserted to convert it to palindrome.
    private int findMinInsertionsDP(String str) {
        int len = str.length();
        int[][] dp = new int[len][len];

        for(int gap = 1; gap < len; gap++) {
            for(int lo = 0, hi = gap; hi < len; lo++, hi++) {
                if(str.charAt(lo) == str.charAt(hi)) {
                    dp[lo][hi] = dp[lo + 1][hi - 1];
                } else {
                    dp[lo][hi] = Math.min(dp[lo + 1][hi], dp[lo][hi - 1]) + 1;
                }
            }
        }
        return dp[0][len - 1];
    }

    // output the length of longest common subsequence
    private int longestCommonSubsequence(String s1, String s2) {
        int len1 = s1.length(), len2 = s2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];
        for(int i = 1; i <= len1; i++) {
            for(int j = 1; j <= len2; j++) {
                if(s1.charAt(i - 1) == s2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        return dp[len1][len2];
    }

    public void reverseList(ListNode head) {
        if (head == null) {
            return;
        }
        ListNode curr = head;
        int length = 0;
        while (curr != null) {//get the total length
            curr = curr.next;
            length++;
        }
        helper(head, length);
    }

    private void helper(ListNode head, int length) {
        if (length == 1) {
            System.out.print(head.val + " ");
            return;//remember to return !!!
        }
        ListNode curr = head;
        int count = 0;
        while (count < length / 2) {
            curr = curr.next;
            count++;
        }
        helper(curr, length - length / 2);//note that the right part has length - length / 2 nodes
        helper(head, length / 2);
    }

    // find longest Arithmetic Progression subsequence
    int lenghtOfLongestAP(int nums[]) {
        int len = nums.length;
        if(len <= 2) return len;
        // Create a table and initialize all values as 2. The value of
        // L[i][j] stores LLAP with set[i] and set[j] as first two
        // elements of AP. Only valid entries are the entries where j>i
        int[][] dp = new int[len][len];
        int res = 2;

        // Fill entries in last column as 2. There will always be
        // two elements in AP with last number of set as second
        // element in AP
        for (int i = 0; i < len; i++) {
            dp[i][len - 1] = 2;
        }
        // Consider every element as second element of AP
        for (int j= len - 2; j >= 1; j--) {
            // Search for i and k for j
            int i = j - 1, k = j + 1;
            while (i >= 0 && k <= len - 1) {
                if (nums[i] + nums[k] < 2 * nums[j]) {
                    k++;
                } else if(nums[i] + nums[k] > 2 * nums[j]) {
                    dp[i][j] = 2;
                    i--;
                } else {
                    // Found i and k for j, LLAP with i and j as first two
                    // elements is equal to LLAP with j and k as first two
                    // elements plus 1. L[j][k] must have been filled
                    // before as we run the loop from right side
                    dp[i][j] = dp[j][k] + 1;
                    // Update overall LLAP, if needed
                    res = Math.max(res, dp[i][j]);
                    // Change i and k to fill more L[i][j] values for
                    // current j
                    i--; k++;
                }
            }
            // If the loop was stopped due to k becoming more than
            // n-1, set the remaining entties in column j as 2
            while (i >= 0)
            {
                dp[i][j] = 2;
                i--;
            }
        }
        return res;
    }

    int numOfAP(int nums[]) {
        return Integer.MAX_VALUE;
    }

    private void findFirstNonMatchLeaf(TreeNode root1, TreeNode root2) {
        Stack<TreeNode> stack1 = new Stack<>();
        Stack<TreeNode> stack2 = new Stack<>();

        stack1.push(root1);
        stack2.push(root2);
        while(!stack1.isEmpty() || !stack2.isEmpty()) {
            // if one of them has been exhausted
            if(stack1.isEmpty() || stack2.isEmpty()) {
                return;
            }
            // iterate to the next leaf of root1
            TreeNode top1 = stack1.pop();
            top1 = getNextLeaf(top1, stack1);

            TreeNode top2 = stack2.pop();
            top2 = getNextLeaf(top2, stack2);

            if(top1 != null && top2 != null) {
                if(top1.val != top2.val) {
                    System.out.print("find first non-matching leaf with val = " + top1.val + "/" + top2.val);
                    return;
                }
            }
        }
    }

    private TreeNode getNextLeaf(TreeNode node, Stack<TreeNode> stack) {
        while(!isLeaf(node)) {
            if(node.right != null) {
                stack.push(node.right);
            }
            if(node.left != null) {
                stack.push(node.left);
            }
            if(!stack.isEmpty()) {
                node = stack.pop();
            } else {
                return null;
            }
        }
        return node;
    }

    private boolean isLeaf(TreeNode node) {
        if(node.left == null && node.right == null) {
            return true;
        } else {
            return false;
        }
    }

    private TreeNode constructBSTFromPreOrderArray(int[] nums) {
        if(nums == null) return null;
        TreeNode root = new TreeNode(nums[0]);
        for(int i = 1; i < nums.length; i++) {
            construct(root, nums[i]);
        }
        return root;
    }

    private void construct(TreeNode root, int val) {
        if(val < root.val) {
            if(root.left == null) {
                root.left = new TreeNode(val);
            } else {
                construct(root.left, val);
            }
        }
        if(val > root.val) {
            if(root.right == null) {
                root.right = new TreeNode(val);
            } else {
                construct(root.right, val);
            }
        }
    }

    private TreeNode constructBSTFromPostOrderArray(int[] nums) {
        if(nums == null) return null;
        TreeNode root = new TreeNode(nums[nums.length - 1]);
        for(int i = nums.length - 2; i >= 0; i--) {
            constructPost(root, nums[i]);
        }
        return root;
    }

    private void constructPost(TreeNode root, int val) {
        if(val > root.val) {
            if(root.right == null) {
                root.right = new TreeNode(val);
            } else {
                constructPost(root.right, val);
            }
        }
        if(val < root.val) {
            if(root.left == null) {
                root.left = new TreeNode(val);
            } else {
                constructPost(root.left, val);
            }
        }
    }

    public List<Integer> subsets(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        helper(res, nums, 1, 0);
        Collections.sort(res); //if res should be sorted, add this
        return res;
    }

    private void helper(List<Integer> res, int[] nums, int product, int index) {
        //if we only need to print the results, we don't need res, we can use this:
        // if (product != 1) {
        //     System.out.print(product + " ");
        // }
        if (product != 1) {
            res.add(product);
        }
        for (int i = index; i < nums.length; i++) {
            product *= nums[i];
            helper(res, nums, product, i + 1);
            product /= nums[i];
        }
    }

    private List<Integer> subsetsIter(int[] nums) {
        List<Integer> res = new ArrayList<>();
        if (nums == null || nums.length == 0) {
            return res;
        }
        int n = nums.length;
        //Arrays.sort(nums); no need to sort
        for (int i = 0; i < (1 << n); i++) {//2^n kinds of state
            int product = 1;
            for (int j = 0; j < n; j++) {//if jth bit of i is 1,nums[j] exists in this combination;Note set doesn't have order
                if ((i & (1 << j)) != 0) {//don't use ==1 cuz this's only jth bit! eg.110 & 010=010=2,doesn't means it's 1 !!
                    product *= nums[j];//also remember to add () for (i & (1 << j)) !!!!!!
                    System.out.println("i = " + i + "   j = " + j);
                }
            }
            res.add(new Integer(product));
        }
        return res;
    }

    static final int NO_OF_CHARS = 256;
    // finds the second most frequently occurring
    // char
    static char getSecondMostFreq(String str)
    {
        // count number of occurrences of every
        // character.
        int[] count = new int[NO_OF_CHARS];
        int i;
        for (i=0; i< str.length(); i++)
            (count[str.charAt(i)])++;

        // Traverse through the count[] and find
        // second highest element.
        int first = 0, second = 0;
        for (i = 0; i < NO_OF_CHARS; i++)
        {
            /* If current element is smaller than
            first then update both first and second */
            if (count[i] > count[first])
            {
                second = first;
                first = i;
            }

            /* If count[i] is in between first and
            second then update second  */
            else if (count[i] > count[second])
                second = i;
        }

        return (char)second;
    }

    // Give numbers from 0 to N - 1 with different weight, return a random number according to their weight
    private int randomAccordingWeight(int[] weight) {
        if(weight == null || weight.length == 0) return -1;

        int sum = 0, len = weight.length;
        int[] accumulated = new int[len];
        for(int i = 0; i < len; i++) {
            sum += weight[0];
            accumulated[i] = sum;
        }

        Random random = new Random();
        int pivot = random.nextInt(accumulated[len - 1]);
        return binarySearch(weight, pivot);
    }

    // find the first appearance of a element which is greater than target
    private int binarySearch(int[] weight, int target) {
        int lo = 0, hi = weight.length;
        while(lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if(weight[mid] <= target) {
                // lo can be smaller than target or the right pos
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    private ListNode flattenII(ListNode head) {
        if(head == null) return null;

        ListNode next = head.next, p = head;
        head.next = null;
        if(p.down != null) {
            p.next = flattenII(head.down);
            p.down = null;
        }
        while (p.next != null) {
            p = p.next;
        }
        if(next != null) {
            p.next = flattenII(next);
        }
        return head;
    }

    private ListNode tail;
    public ListNode flatten(ListNode head) {
        if (head == null) {
            return null;
        }
        tail = head;//keep track of the tail of list
        ListNode next = head.next;//store the next

        if (head.down != null) {
            head.next = flatten(head.down);//connect all nodes in curr head's down part
        }
        if (next != null) {
            tail.next = flatten(next);//connect the tail to the next part
        }
        return head;
    }

    // assume head is not null
    private ListNode levelFlatten(ListNode head) {
        ListNode tail = head;
        Queue<ListNode> queue = new LinkedList<>();
        ListNode prev = null;
        while(tail != null) {
            if(tail.down != null) {
                queue.add(tail.down);
            }
            prev = tail;
            tail = tail.next;
        }
        tail = prev;
        while(!queue.isEmpty()) {
            ListNode node = queue.poll();
            tail.next = node;
            tail = tail.next;
            while(tail != null) {
                if(tail.down != null) {
                    queue.add(tail.down);
                }
                prev = tail;
                tail = tail.next;
            }
            tail = prev;
        }
        return head;
    }

    public void printLinkedList (ListNode head) {
        if(head == null) return ;

        ListNode p = head;
        while(p != null) {
            System.out.print(p.val + "  ");
            p = p.next;
        }

    }

    private int longestAP(int[] nums) {
        // inner map: d -> AP len
        int len = nums.length;
        Map<Integer, Integer>[] dToDiff = new HashMap[len];
        int maxLen = 2;
        for(int i = len - 1; i >= 0; i--) {
            dToDiff[i] = new HashMap<>();
            for(int j = i + 1; j < len; j++) {
                int d = nums[j] - nums[i];
                int tmp = dToDiff[j].getOrDefault(d, 1) + 1;
                if(!dToDiff[i].containsKey(d) || dToDiff[i].get(d) < tmp) {
                    dToDiff[i].put(d, tmp);
                    maxLen = Math.max(maxLen, dToDiff[i].get(d));
                }
            }
        }
        return maxLen;
    }

    private int numOfAPII(int[] nums) {
        int len = nums.length;
        Map<Integer, Integer>[] dToNums = new HashMap[len];
        int res = 0;
        for(int i = 0; i < len; i++) {
            dToNums[i] = new HashMap<>();
            for(int j = 0; j < i; j++) {
                int d = nums[i] - nums[j];
                int num_j_d = dToNums[j].getOrDefault(d, 0);
                int num_i_d = num_j_d + 1;
                res += num_i_d;
                dToNums[i].put(d, num_i_d);
            }
        }
        return res - len * (len - 1) /2 ;
    }

    private DLLNode DLLToBalancedBST(DLLNode root) {
        if(root == null) return null;
        int len = 0;
        DLLNode p = root;
        while(p != null) {
            p = p.right;
            len++;
        }
        return DLLTOBalancedBSTHelper(root, len);
    }

    private DLLNode DLLTOBalancedBSTHelper(DLLNode start, int size) {
        if(size <= 0) {
            return null;
        }
        DLLNode p = start;
        int idx = 0;
        // this statemenet make p the node next to size/2 node
        while(idx++ < size / 2) {
            p = p.right;
        }
        p.left = DLLTOBalancedBSTHelper(start, size / 2);
        p.right = DLLTOBalancedBSTHelper(p.right, size - size / 2 - 1);
        return p;
    }

    DLLNode head;
    private DLLNode DLLToBalancedBSTLinearTime(int size) {
        if(size <= 0) return null;

        DLLNode left = DLLToBalancedBSTLinearTime(size / 2);

        DLLNode root = head;
        root.left = left;
        head = head.right;

        root.right = DLLToBalancedBSTLinearTime(size - size / 2 - 1);
        return root;
    }

    private int countPalindromicSubsquence(String str) {
        int len = str.length();
        int[][] dp = new int[len][len];
        for(int i = 0; i < len; i++) {
            dp[i][i] = 1;
        }
        for(int gap = 1; gap < len; gap++) {
            for(int i = 0, j = gap; j < len; i++, j++) {
                if(str.charAt(i) == str.charAt(j)) {
                    dp[i][j] = dp[i][j - 1] + dp[i + 1][j] + 1;
                } else {
                    dp[i][j] = dp[i][j - 1] + dp[i + 1][j] - dp[i + 1][j - 1];
                }
            }
        }
        return dp[0][len - 1];
    }

    // setting: randomly pick k pos in h*w array as bomb
    // we assume that k is less than h * w
    private int[][] bomb(int h, int w, int k) {
        int[][] res = new int[h][w];
        // initialization
        int[] pos = new int[k];
        for(int i = 0; i < k; i++) {
            int x = i / w;
            int y = i % w;
            res[x][y] = 1;
            pos[i] = i;
        }
        // reservoir sampling
        Random random = new Random();
        for(int i = k; i < h * w; i++) {
            int rand = random.nextInt(i + 1);
            if(rand < k) {
                int x = pos[rand] / w;
                int y = pos[rand] % w;
                pos[rand] = i;
                res[x][y] = 0;
                x = i / w;
                y = i % w;
                res[x][y] = 1;
            }
        }
        return res;
    }

    public String decodeString(String s) {
        Stack<Integer> intStack = new Stack<>();
        Stack<StringBuilder> strStack = new Stack<>();
        StringBuilder cur = new StringBuilder();
        int k = 0;
        for (char ch : s.toCharArray()) {
            if (Character.isDigit(ch)) {
                k = k * 10 + ch - '0';
            } else if ( ch == '[') {
                intStack.push(k);
                strStack.push(cur);
                cur = new StringBuilder();
                k = 0;
            } else if (ch == ']') {
                StringBuilder tmp = cur;
                cur = strStack.pop();
                for (k = intStack.pop(); k > 0; --k) cur.append(tmp);
            } else {
                cur.append(ch);
            }
        }
        return cur.toString();
    }

    public List<String> wordBreak(String s, List<String> wordDict) {
        int maxlen = 0;
        Set<String> words = new HashSet<>();
        // add word in set for quick lookup
        for(String word : wordDict) {
            words.add(word);
            maxlen = Math.max(maxlen, word.length());
        }

        List<Integer>[] prevIdx = new ArrayList[s.length() + 1];
        // init prevIdx
        for(int i = 0; i <= s.length(); i++) {
            prevIdx[i] = new ArrayList<>();
        }
        prevIdx[0].add(-1);

        for(int i = 0; i <= s.length(); i++) {
            int j = Math.max(0, i - maxlen);
            for(; j < i; j++) {
                if(prevIdx[j].size() > 0 && words.contains(s.substring(j, i))) {
                    prevIdx[i].add(j);
                }
            }
        }
        List<String> res = new ArrayList<>();
        // BFS
        Queue<String> strs = new LinkedList<>();
        Queue<Integer> idxes = new LinkedList<>();
        strs.add(""); idxes.add(s.length());
        while(!idxes.isEmpty()) {
            int idx = idxes.poll();
            String str = strs.poll();
            List<Integer> prev = prevIdx[idx];
            for(int from : prev) {
                String newStr = new String(s.substring(from, idx) + " " + str);
                if(from == 0) {
                    res.add(newStr.trim());
                } else {
                    strs.add(newStr);
                    idxes.add(from);
                }
            }
        }
        return res;
    }

    public static void main(String[] args) {
        Facebook fb = new Facebook();

        System.out.println(fb.numOfAPII(new int[]{2, 4, 6, 8, 10}));

        ListNode l1 = new ListNode(1);
        ListNode l2 = new ListNode(2);
        ListNode l3 = new ListNode(3);
        ListNode l4 = new ListNode(4);

        ListNode l7 = new ListNode(7);
        ListNode l8 = new ListNode(8);
        ListNode l9 = new ListNode(9);
        ListNode l14 = new ListNode(14);
        ListNode l15 = new ListNode(15);
        ListNode l16 = new ListNode(16);
        ListNode l10 = new ListNode(10);

        l1.next = l2;
        l2.next = l3;
        l3.next = l4;
        l2.down = l7;
        l7.down = l9;
        l9.down = l14;
        l14.down = l15;
        l7.next = l8;
        l8.down = l16;
        l8.next = l10;

        DLLNode node1 = new DLLNode(1);
        DLLNode node2 = new DLLNode(2);
        DLLNode node3 = new DLLNode(3);
        DLLNode node4 = new DLLNode(4);
        DLLNode node5 = new DLLNode(5);
        DLLNode node6 = new DLLNode(6);
        DLLNode node7 = new DLLNode(7);
        node1.right = node2;
        node2.left = node1; node2.right = node3;
        node3.left = node2; node3.right = node4;
        node4.left = node3; node4.right = node5;
        node5.left = node4; node5.right = node6;
        node6.left = node5; node6.right = node7;
        node7.left = node6;
        fb.head = node1;
        //fb.DLLToBalancedBSTLinearTime(7);

//        fb.head = node1;
//        fb.DLLToBalancedBSTLinearTime(7);
//
//        fb.levelFlatten(l1);
//        fb.printLinkedList(l1);
//
//        ListNode head = fb.flattenII(l1);
//        ListNode p = head;
//        while(p != null) {
//            System.out.print(p.val + " ->");
//            p = p.next;
//        }
//        int[] a = new int[]{2,2,2,2,2,2,22};
//        StringBuilder sb = new StringBuilder();
//        System.out.println(fb.wordBreak("catsanddog", Arrays.asList("cat", "cats", "and", "sand", "dog")));
    }
}

//int[] a = new int[]{1, 7, 10, 13, 14, 19};
//System.out.print(fb.lenghtOfLongestAP(a));

/*
TreeNode node1 = new TreeNode(5);
        TreeNode node2 = new TreeNode(2);
        TreeNode node3 = new TreeNode(7);
        TreeNode node4 = new TreeNode(10);
        TreeNode node5 = new TreeNode(11);
        node1.left = node2;
        node1.right = node3;
        node2.left = node4;
        node2.right = node5;

        TreeNode node6 = new TreeNode(6);
        TreeNode node7 = new TreeNode(10);
        TreeNode node8 = new TreeNode(15);
        node6.left = node7;
        node6.right = node8;


ListNode l1 = new ListNode(1);
        ListNode l2 = new ListNode(2);
        ListNode l3 = new ListNode(3);
        ListNode l4 = new ListNode(4);
        ListNode l5 = new ListNode(5);
        ListNode l6 = new ListNode(6);

        l1.next = l2;
        l2.next = l3;
        l3.next = l4;
        l4.next = l5;
        l5.next = l6;*/
