import java.util.Stack;

class MinStack {

    Stack<Long> stack;
    long min;
    /** initialize your data structure here. */
    public MinStack() {
        this.stack = new Stack<>();
    }

    public void push(int x) {
        if(stack.isEmpty()) {
            stack.push(0L);
            min = x;
        } else {
            stack.push((long)x - min);
            if((long)x < min) {
                min = (long)x;
            }
        }
    }

    public void pop() {
        if(stack.peek() < 0) {
            min = min - stack.peek();
        }
        stack.pop();
    }

    public int top() {
        if(stack.peek() < 0) {
            return (int)min;
        } else {
            long res = min + stack.peek();
            return (int)res;
        }
    }

    public int getMin() {
        return (int)min;
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */