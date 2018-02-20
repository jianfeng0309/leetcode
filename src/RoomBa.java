import java.util.*;

public class RoomBa {

    int direction = -1;
    int[] curLocation = new int[]{-1, -1};
    int[][] room = new int[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    Map<String, Set<Integer>> validDirection = new HashMap<>();
    Stack<int[]> trace = new Stack<>();


    Map<Integer, int[]> directions = new HashMap<>();

    public RoomBa(int x, int y, int direction) {
        this.curLocation = new int[]{x, y};
        this.direction = direction;

        init();
    }

    private void init(){
        directions.put(0, new int[]{0, 1});
        directions.put(1, new int[]{1, 0});
        directions.put(2, new int[]{0, -1});
        directions.put(3, new int[]{-1, 0});
    }

    private boolean move() {
        int[] dir = directions.get(direction);
        for(int i = 0; i < 2; i++) {
            curLocation[i] += dir[i];
        }

        int x = curLocation[0], y = curLocation[1];
        if(x >= 0 && x < room.length && y >= 0 && y < room[0].length) {
            return true;
        } else {
            for(int i = 0; i < 2; i++) {
                curLocation[i] -= dir[i];
            }
            return false;
        }
    }

    private void turnLeft(int k) {
    }

    private void turnRight(int k) {

    }

    private void clean() {
        System.out.println("x = " + curLocation[0] + " y = " + curLocation[1] + "  cleaned");
    }

    void changeDirection(int target) {
        while(direction != target){
            direction = (direction + 1) % 4;
            this.turnLeft(1);
        }
    }

    public void cleanRoom(){
        while(true) {
            String key = curLocation[0] + " " + curLocation[1];
            // first time got there
            if(!validDirection.keySet().contains(key)) {
                Set<Integer> set = new HashSet<>();
                set.add(0); set.add(1); set.add(2); set.add(3);
                validDirection.put(key, set);
                clean();
            }

            Iterator<Integer> iter = validDirection.get(key).iterator();
            if(iter.hasNext()) {
                changeDirection(iter.next());
                validDirection.get(key).remove(direction);
                if(move()) {
                    trace.push(new int[]{curLocation[0], curLocation[1]});
                    System.out.println("go to x = " + curLocation[0] + " y = " + curLocation[1]);
                } else {
                    continue;
                }
            } else {
                //getBack();
                if(!trace.isEmpty()) {
                    int[] lastLocation = trace.pop();
                    getBack(lastLocation, curLocation);
                    curLocation[0] = lastLocation[0];
                    curLocation[1] = lastLocation[1];
                } else {
                    break;
                }
            }
        }
    }

    private void getBack(int[] last, int[] cur){
        int x = last[0] - cur[0];
        int y = last[1] - cur[1];
        int target = -1;
        if(x == 0 && y == 1) {
            target = 0;
        } else if(x == 0 && y == -1) {
            target = 2;
        } else if(x == 1 && y == 0) {
            target = 1;
        } else {
            target = 3;
        }
        changeDirection(target);
        move();
    }

    public static void main (String[] args) {
        RoomBa roomBa = new RoomBa(0, 0, 0);
        roomBa.cleanRoom();

    }
}
