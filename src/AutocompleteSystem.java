import java.util.*;

class AutocompleteSystem {

    class TrieNode {
        // all child nodes of this node
        Map<Character, TrieNode> children;
        // all strings that pass by this node
        Map<String, Integer> counter;
        // whether this is the end of a word
        boolean item;
        public TrieNode() {
            this.children = new HashMap<>();
            this.counter = new HashMap<>();
            this.item = false;
        }
    }

    class Pair {
        String s;
        int count;
        public Pair(String s, int count) {
            this.s = s;
            this.count = count;
        }
    }

    private TrieNode root;
    private String prefix;
    private TrieNode preNode;

    public AutocompleteSystem(String[] sentences, int[] times) {
        root = new TrieNode();
        prefix = "";
        preNode = root;
        for(int i = 0; i < sentences.length; i++) {
            add(sentences[i], times[i]);
        }
    }

    private void add(String sentence, int time) {
        TrieNode node = root;
        for(int i = 0; i < sentence.length(); i++) {
            char c = sentence.charAt(i);
            if(!node.children.containsKey(c)) {
                node.children.put(c, new TrieNode());
            }
            // reach the node
            node = node.children.get(c);
            // update all related fields
            node.counter.put(sentence, node.counter.getOrDefault(sentence, 0) + time);
        }
        node.item = true;
    }

    public List<String> input(char c) {
        if(c == '#') {
            add(prefix, 1);
            prefix = "";
            preNode = root;
            return new ArrayList<>();
        }
        prefix += c;
        if(preNode == null || !preNode.children.containsKey(c)) {
            preNode = null;
            return new ArrayList<>();
        } else {
            preNode = preNode.children.get(c);
            // construct our result
            Queue<Pair> pq = new PriorityQueue<>(new Comparator<Pair>() {
                @Override
                public int compare(Pair o1, Pair o2) {
                    if(o1.count != o2.count) {
                        return o2.count - o1.count;
                    } else {
                        return o1.s.compareTo(o2.s);
                    }
                }
            });

            for(String key : preNode.counter.keySet()) {
                pq.add(new Pair(key, preNode.counter.get(key)));
            }

            List<String> res = new ArrayList<>();
            while(!pq.isEmpty() && res.size() < 3) {
                res.add(pq.poll().s);
            }
            return res;
        }
    }
}

/**
 * Your AutocompleteSystem object will be instantiated and called as such:
 * AutocompleteSystem obj = new AutocompleteSystem(sentences, times);
 * List<String> param_1 = obj.input(c);
 */