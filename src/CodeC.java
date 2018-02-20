import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class CodeC {

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    private static final String nulln = "#";
    private static final String separ = ":";

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        seri(root, sb);
        return sb.toString();
    }

    private void seri(TreeNode node, StringBuilder sb) {
        if(node == null) {
            sb.append(nulln).append(separ);
            return;
        }
        sb.append(node.val).append(separ);
        seri(node.left, sb);
        seri(node.right, sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Queue<String> queue = new LinkedList<>();
        queue.addAll(Arrays.asList(data.split(separ)));
        return deseri(queue);
    }

    private TreeNode deseri(Queue<String> data) {
        String ele = data.poll();
        if(ele.equals("#")) {
            return null;
        } else {
            TreeNode root = new TreeNode(Integer.parseInt(ele));
            root.left = deseri(data);
            root.right = deseri(data);
            return root;
        }
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.deserialize(codec.serialize(root));