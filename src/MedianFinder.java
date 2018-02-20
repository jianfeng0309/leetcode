import java.util.PriorityQueue;
import java.util.Queue;

class MedianFinder {
    Queue<Integer> maxHeap = new PriorityQueue<>();
    Queue<Integer> minHeap = new PriorityQueue<>();

    /** initialize your data structure here. */
    public MedianFinder() {

    }

    public void addNum(int num) {
        maxHeap.add(num);
        minHeap.add(-maxHeap.poll());
        if(minHeap.size() > maxHeap.size()) {
            maxHeap.add(-minHeap.poll());
        }
    }

    public double findMedian() {
        if(maxHeap.size() > minHeap.size()) {
            return (double)maxHeap.peek();
        } else {
            return ((double)maxHeap.peek() - (double)minHeap.peek()) / 2.0;
        }
    }

    public static void main(String[] args) {
        MedianFinder medianFinder = new MedianFinder();
        medianFinder.addNum(1);
        medianFinder.addNum(2);
        System.out.println(medianFinder.findMedian());
    }
}