import com.sun.org.apache.xerces.internal.util.SymbolHash;

import java.util.HashMap;
import java.util.Map;

class LRUCache {
    class Node {
        Node prev = null, next = null;
        int key, val;
        public Node (int key, int val) {
            this.key = key;
            this.val = val;
        }
    }

    Node dummyHead = new Node(Integer.MIN_VALUE, Integer.MIN_VALUE),
            dummyTail = new Node(Integer.MAX_VALUE, Integer.MAX_VALUE);

    private int capacity = 0, size = 0;
    Map<Integer, Node> map = new HashMap<>();

    public LRUCache(int capacity) {
        dummyHead.next = dummyTail;
        dummyTail.prev = dummyHead;
        this.capacity = capacity;
    }

    public int get(int key) {
        if(map.containsKey(key)) {
            Node node = map.get(key);
            removeNode(node);
            appendFirst(node);
            return node.val;
        } else {
            return - 1;
        }
    }

    private void appendFirst(Node node) {
        map.put(node.key, node);
        Node ori = dummyHead.next;
        node.prev = dummyHead;
        node.next = ori;
        dummyHead.next = node;
        ori.prev = node;
    }

    private void evictLast(){
        Node last = dummyTail.prev;
        map.remove(last.key);
        last.prev.next = last.next;
        last.next.prev = last.prev;
    }

    private void removeNode(Node node) {
        map.remove(node.key);
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    public void put(int key, int value) {
        Node newNode = new Node(key, value);
        if(size == capacity) {
            if(map.containsKey(key)) {
                Node node = map.get(key);
                removeNode(node);
                appendFirst(newNode);
            } else {
                appendFirst(newNode);
                evictLast();
            }
        } else {
            if(map.containsKey(key)) {
                Node node = map.get(key);
                removeNode(node);
                appendFirst(newNode);
            } else {
                size++;
                appendFirst(newNode);
            }
        }
    }


}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */