/**
 * Created by GuoJianFeng on 2/23/18.
 */
public class TicTacToeAI {


    public boolean move(TicTacToe ticTacToe) throws Exception{
        int[] pos = ticTacToe.checkFull();
        ticTacToe.add(pos[0], pos[1], 'O');
        return true;
    }
}
