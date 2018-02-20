import java.lang.reflect.Array;
import java.util.*;

/**
 * Created by GuoJianFeng on 9/18/17.
 */
public class Main {
    int size = 0;
    public String alienOrder(String[] words) {
        Map<Character, Integer> inDegree = new HashMap<>();
        Map<Character, List<Character>> adj = new HashMap<>();
        initGraph(inDegree, adj, words);
        return topoSort(inDegree, adj);
    }

    private void initGraph(Map<Character, Integer> inDegree, Map<Character, List<Character>> adj, String[] words) {
        // add all unique char in set
        HashSet<Character> set = new HashSet<>();
        for(int i = 0; i < words.length; i++) {
            for(int j = 0; j < words[i].length(); j++) {
                set.add(words[i].charAt(j));
            }
        }
        this.size = set.size();
        for(char c : set) {
            adj.put(c, new ArrayList<>());
            inDegree.put(c, 0);
        }
        for(int i = 0; i < words.length - 1; i++) {
            String s1 = words[i], s2 = words[i + 1];
            int len = Math.min(s1.length(), s2.length());
            for(int j = 0; j < len; j++) {
                if(s1.charAt(j) != s2.charAt(j)) {
                    char from = s1.charAt(j), to = s2.charAt(j);
                    inDegree.put(to, inDegree.get(to) + 1);
                    adj.get(from).add(to);
                    break;
                }
            }
        }
    }

    private String topoSort(Map<Character, Integer> inDegree, Map<Character, List<Character>> adj) {
        Queue<Character> queue = new LinkedList<>();
        for(char key : inDegree.keySet()) {
            if(inDegree.get(key) == 0) {
                queue.add(key);
            }
        }
        StringBuilder sb = new StringBuilder();
        while(!queue.isEmpty()) {
            char element = queue.poll();
            sb.append(element);
            for(char neigh : adj.get(element)) {
                int degree = inDegree.get(neigh) - 1;
                if(degree == 0) {
                    queue.add(neigh);
                }
                inDegree.put(neigh, degree);
            }
        }
        if(sb.length() == this.size) {
            return sb.toString();
        } else {
            return "";
        }
    }

    public static void main(String[] args) {
        Main main = new Main();
        String[] words = {"wrt", "wrf", "er", "ett", "rftt"};
        System.out.println(main.alienOrder(words));
    }
}
