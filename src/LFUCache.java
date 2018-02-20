import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;

/**
 * Created by GuoJianFeng on 12/6/17.
 */
public class LFUCache {

    class Node {
        int count;
        LinkedHashSet<Integer> keys;
        Node prev = null;
        Node next = null;
        public Node (int count) {
            this.count = count;
        }
    }

    Node head = null;
    Map<Integer, Integer> valueHash = new HashMap<>();
    Map<Integer, Node> nodeHash = new HashMap<>();
    int capacity = 0;

    public LFUCache(int capacity) {
        this.capacity = capacity;
    }

    public int get(int key) {
        if(valueHash.containsKey(key)) {
            increasingCount(key);
            return valueHash.get(key);
        } else {
            return -1;
        }
    }

    public void put(int key, int value) {
        if(capacity == 0) return;

        if(valueHash.containsKey(key)) {
            valueHash.put(key, value);
        } else {
            if(valueHash.size() < capacity) {
                valueHash.put(key, value);
            } else {
                removeOld();
                valueHash.put(key, value);
            }
            addToHead(key);
        }
        increasingCount(key);
    }

    private void addToHead(int key) {
        if(head == null) {
            head = new Node(0);
            head.keys.add(key);
        } else if (head.count > 0) {
            Node node = new Node(0);
            node.keys.add(key);
            node.next = head;
            head.prev = node;
            head = node;
        } else {
            head.keys.add(key);
        }
        nodeHash.put(key, head);
    }

    private void increasingCount(int key) {
        Node node = nodeHash.get(key);
        node.keys.remove(key);

        if(node.next == null) {
            Node next = new Node(node.count + 1);
            node.next = next;
            next.prev = node;
            next.keys.add(key);
        } else if(node.next.count == node.count + 1) {
            node.next.keys.add(key);
        } else  {
            Node next = new Node(node.count + 1);
            next.keys.add(key);
            next.next = node.next;
            next.prev = node;
            node.next.prev = next;
            node.next = next;
        }
        nodeHash.put(key, node.next);
        if(node.keys.size() == 0) {
            remove(node);
        }
    }

    private void removeOld() {
        if (head == null) return;
        int old = 0;
        for (int n: head.keys) {
            old = n;
            break;
        }
        head.keys.remove(old);
        if (head.keys.size() == 0) remove(head);
        nodeHash.remove(old);
        valueHash.remove(old);
    }

    private void remove(Node node) {
        if (node.prev == null) {
            head = node.next;
        } else {
            node.prev.next = node.next;
        }
        if (node.next != null) {
            node.next.prev = node.prev;
        }
    }
}
