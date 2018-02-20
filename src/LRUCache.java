import com.sun.org.apache.xerces.internal.util.SymbolHash;

import java.util.HashMap;

class LRUCache {
    class DoubleLinkedList {
        int val;
        int key;
        DoubleLinkedList prev = null;
        DoubleLinkedList next = null;
        public DoubleLinkedList(int key, int val) {
            this.key = key;
            this.val = val;
        }
    }

    int size = 0;
    int capacity;
    HashMap<Integer, DoubleLinkedList> map = new HashMap<>();
    DoubleLinkedList dommy = new DoubleLinkedList(0, -1);
    DoubleLinkedList dommyTail = new DoubleLinkedList(0, -1);

    public LRUCache(int capacity) {
        this.capacity = capacity;
        dommy.next = dommyTail;
        dommyTail.prev = dommy;
    }

    public int get(int key) {
        if(this.capacity == 0) return -1;
        if(map.containsKey(key)) {
            DoubleLinkedList ele = map.get(key);
            remove(ele);
            appendAndUpdate(key, ele);
            return map.get(key).val;
        } else {
            return -1;
        }
    }

    public void put(int key, int value) {
        if(capacity == 0) return;

        // size vs capacity
        if(size == capacity) {
            if(map.containsKey(key)) {
                // no need to delete tail
                DoubleLinkedList ele = map.get(key);
                remove(ele);
                DoubleLinkedList newElement = new DoubleLinkedList(key, value);
                appendAndUpdate(key, newElement);
            } else {
                // delete tail, append head
                DoubleLinkedList newElement = new DoubleLinkedList(key, value);
                appendAndUpdate(key, newElement);
                deleteTail();
            }

        } else {
            // key exist or not
            if(map.containsKey(key)) {
                // remove element, update and append head
                DoubleLinkedList ele = map.get(key);
                remove(ele);
                DoubleLinkedList newElement = new DoubleLinkedList(key, value);
                appendAndUpdate(key, newElement);
            } else {
                // just append head and size++;
                DoubleLinkedList newElement = new DoubleLinkedList(key, value);
                appendAndUpdate(key, newElement);
                size += 1;
            }
        }
    }

    private void remove(DoubleLinkedList ele) {
        DoubleLinkedList prev = ele.prev, next = ele.next;
        prev.next = next;
        next.prev = prev;
    }

    private void appendAndUpdate(int key, DoubleLinkedList ele) {
        map.put(key, ele);
        DoubleLinkedList head = dommy.next;
        ele.prev = dommy;
        ele.next = head;
        head.prev = ele;
        dommy.next = ele;
    }

    private void deleteTail() {
        DoubleLinkedList tail = dommyTail.prev, tailPrev = tail.prev;
        map.remove(tail.key);
        tailPrev.next = dommyTail;
        dommyTail.prev = tailPrev;
    }

    public static void main(String[] args) {
        LRUCache lruCache = new LRUCache(2);
        lruCache.put(1, 1);
        lruCache.put(2, 2);
        System.out.println(lruCache.get(1));
        lruCache.put(3, 3);
        System.out.println(lruCache.get(2));
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */