import java.util.Scanner;

/**
 * Created by GuoJianFeng on 2/23/18.
 */
public class TicTacToe {

    private char[][] board = new char[3][3];

    public char[][] getBoard(){
        return this.board;
    }

    public TicTacToe() {
        for(int i = 0; i < board.length; i++) {
            for(int j = 0; j < board[0].length; j++) {
                board[i][j] = '-';
            }
        }
    }

    public boolean add(int row, int col, char token) throws Exception {
        if(board[row][col] == '-') {
            board[row][col] = token;
            return true;
        } else {
            return false;
        }
    }

    public void print() {
        for(int i = 0; i < board.length; i++) {
            System.out.println(board[i][0] + " | " + board[i][1] + " | " + board[i][2]);
        }
    }

    public int[] checkFull() {
        for(int i = 0; i < board.length; i++) {
            for(int j = 0; j < board[0].length; j++) {
                if(this.board[i][j] == '-') {
                    return new int[]{i, j};
                }
            }
        }
        return new int[]{- 1, -1};
    }

    public boolean humanMove(int row, int col) throws Exception {
        return this.add(row, col, 'X');
    }

    public static void main(String[] args) throws Exception{
        TicTacToe ticTacToe = new TicTacToe();
        TicTacToeAI ai = new TicTacToeAI();
        Scanner scanner = new Scanner(System.in);

        int step = 9;
        boolean human = true;
        while(step > 0) {
            ticTacToe.print();
            if(human) {
                //to do
                System.out.println("Please enter the pos you wanna to add with whitespace separated (example like " + "2, 2)");

                String pos = scanner.nextLine();
                String[] tmp = pos.split(" ");
                int row = Integer.parseInt(tmp[0]);
                int col = Integer.parseInt(tmp[1]);
                if(ticTacToe.humanMove(row, col)) {
                    human = false;
                    step--;
                } else {
                    continue;
                }
            } else {
                ai.move(ticTacToe);
                human = true;
                step--;
            }
        }
    }
}
