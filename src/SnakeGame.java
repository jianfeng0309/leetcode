import java.util.Map;

class SnakeGame {

    /*class Node {
        int x;
        int y;
        Node prev,next;
        public Node(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    int[][] board;
    int score;
    int width, height;

    int[][] food;
    int foodIdx;

    Node dummyHead = new Node(-1, -1), dummyTail = new Node(-1, -1);

    private  Map<String, Integer> map = new HashMap<>();
    private int[] xx = {-1, 0, 0, 1};
    private int[] yy = {0, -1, 1, 0};
    *//** Initialize your data structure here.
     @param width - screen width
     @param height - screen height
     @param food - A list of food positions
     E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0]. *//*
    public SnakeGame(int width, int height, int[][] food) {
        map.put("U", 0); map.put("L", 1); map.put("R", 2);map.put("D", 3);
        this.width = width;
        this.height = height;
        this.board = new int[height][width];
        this.food = food;
        score = 0;

        foodIdx = 0;
        if(food != null && foodIdx < food.length) {
            board[food[foodIdx][0]][food[foodIdx][1]] = 2;
        }

        board[0][0] = 1;
        Node snake = new Node(0, 0);
        dummyHead.next = snake;
        snake.prev = dummyHead;
        snake.next = dummyTail;
        dummyTail.prev = snake;

    }

    *//** Moves the snake.
     @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down
     @return The game's score after the move. Return -1 if game over.
     Game over when snake crosses the screen boundary or bites its body. *//*
    public int move(String direction) {
        int x = dummyHead.next.x + xx[map.get(direction)], y = dummyHead.next.y + yy[map.get(direction)];
        if(x >= 0 && x < height && y >= 0 && y < width) {
            if(board[x][y] == 2) {
                board[x][y] = 1;
                appendFirst(new Node(x, y));

                foodIdx += 1;
                if(food != null && foodIdx < food.length) {
                    board[food[foodIdx][0]][food[foodIdx][1]] = 2;
                }

                score += 1;
                return score;
            }  else {
                //board[x][y] = 1;
                appendFirst(new Node(x, y));
                board[dummyTail.prev.x][dummyTail.prev.y] = 0;
                deleteLast();
                if(board[x][y] == 1) {
                    return -1;
                } else {
                    board[x][y] = 1;
                }
                return score;
            }
        } else {
            return -1;
        }
    }

    private void appendFirst(Node node) {
        Node ori = dummyHead.next;
        node.next = ori;
        node.prev = dummyHead;
        ori.prev = node;
        dummyHead.next = node;
    }

    private void deleteLast() {
        Node last = dummyTail.prev;
        last.prev.next = last.next;
        last.next.prev = last.prev;
    }*/

}