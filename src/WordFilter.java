import java.util.*;

class WordFilter {

    class TrieNode {
        Set<String> words = new HashSet<>();
        TrieNode[] children = new TrieNode[26];
    }

    TrieNode prefixRoot = new TrieNode();
    TrieNode suffixRoot = new TrieNode();
    Map<String, Integer> map = new HashMap<>();

    private void addWord(String word, boolean prefix) {
        TrieNode p = prefixRoot;
        if(!prefix){
            p = suffixRoot;
        }
        p.words.add(prefix? word : new StringBuilder(word).reverse().toString());
        for(char c : word.toCharArray()) {
            if(p.children[c - 'a'] == null) {
                p.children[c - 'a'] = new TrieNode();
            }
            p = p.children[c - 'a'];
            p.words.add(prefix? word : new StringBuilder(word).reverse().toString());
        }
    }

    public WordFilter(String[] words) {
        for(int i = 0; i < words.length; i++) {
            String word = words[i];
            String reverse = new StringBuilder(word).reverse().toString();
            addWord(word, true);
            addWord(reverse, false);
            map.put(word, i);
        }

    }

    public int f(String prefix, String suffix) {
        Set<String> forward = new HashSet<>();
        Set<String> backward = new HashSet<>();

        TrieNode p = prefixRoot;
        for(char c : prefix.toCharArray()) {
            if(p.children[c - 'a'] != null) {
                p = p.children[c - 'a'];
            } else {
                p = null;
                break;
            }
        }
        if(p != null) {
            forward = p.words;
        }

        p = suffixRoot;
        String suf = new StringBuilder(suffix).reverse().toString();
        for(char c : suf.toCharArray()) {
            if(p.children[c - 'a'] != null) {
                p = p = p.children[c - 'a'];
            } else {
                p = null;
                break;
            }
        }
        if(p != null) {
            backward = p.words;
        }

        int max = -1;
        for(String word : forward) {
            if(backward.contains(word)) {
                max = Math.max(max, map.get(word));
            }
        }
        return max;
    }

    public static void main(String[] args) {
        String[] words = {"abbbababbb","baaabbabbb","abababbaaa","abbbbbbbba","bbbaabbbaa","ababbaabaa","baaaaabbbb","babbabbabb","ababaababb","bbabbababa"};
        WordFilter wordFilter = new WordFilter(words);
        System.out.println(wordFilter.f("babbab",""));
    }
}

/**
 * Your WordFilter object will be instantiated and called as such:
 * WordFilter obj = new WordFilter(words);
 * int param_1 = obj.f(prefix,suffix);
 */