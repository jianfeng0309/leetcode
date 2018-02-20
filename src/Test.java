import java.util.TreeMap;

/**
 * Created by GuoJianFeng on 12/5/17.
 */
public class Test {

    static class ListNode {
        int val;
        ListNode next;
        public ListNode(int val) {
            this.val = val;
        }
    }

    static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        public TreeNode(int val) {
            this.val = val;
        }
    }

    private TreeNode sortedListNodeToBST(ListNode head) {
        int size = 0;
        ListNode p = head;
        while(p != null) {
            size++;
            p = p.next;
        }
        return sortedListNodeToBSTHelper(head, size);
    }

    private TreeNode sortedListNodeToBSTHelper(ListNode head, int size) {
        if(size <= 0) return null;

        ListNode p = head;
        int idx = 0;
        while(idx++ < size / 2) {
            p = p.next;
        }
        TreeNode root = new TreeNode(p.val);
        root.left = sortedListNodeToBSTHelper(head, size/ 2);
        root.right = sortedListNodeToBSTHelper(p.next, size - size / 2 - 1);
        return root;
    }

    ListNode head;
    private TreeNode sortedListToTreeLinear(int size) {
        if(size <= 0) return null;

        TreeNode left = sortedListToTreeLinear(size / 2);
        TreeNode root = new TreeNode(head.val);
        head = head.next;
        root.left = left;
        root.right = sortedListToTreeLinear(size - 1 - size / 2);
        return root;
    }

    public static void main(String[] args) {
        ListNode l1 = new ListNode(1);
        ListNode l2 = new ListNode(2);
        ListNode l3 = new ListNode(3);
        ListNode l4 = new ListNode(4);
        ListNode l5 = new ListNode(5);
        l1.next = l2; l2.next = l3;
        l3.next = l4; l4.next = l5;
        Test test = new Test();
        test.head = l1;
        TreeNode node = test.sortedListToTreeLinear(5);

    }
}
