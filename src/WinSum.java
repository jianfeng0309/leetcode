import java.util.ArrayList;
import java.util.List;

/**
 * Created by GuoJianFeng on 9/30/17.
 */
public class WinSum {

    public List<Integer> getSum(List<Integer> list, int k) {
        List<Integer> res = new ArrayList<>();
        if(list == null || list.size() == 0 || k > list.size()) {
            return res;
        }

        //get first k element in list, add to the res as the first element of res
        int sum = 0;
        for(int i = 0; i < k; i++) {
            sum += list.get(i);
        }
        res.add(sum);
        int size = list.size();
        for(int i = k; i < size; i++) {
            sum = sum + list.get(i) - list.get(i - k);
            res.add(sum);
        }
        return res;
    }

    public static void main(String[] args) {
        WinSum winSum = new WinSum();
        List<Integer> list = new ArrayList<>();
        list.add(1); list.add(2); list.add(3); list.add(4); list.add(5);
        List<Integer> res = winSum.getSum(list, 2);
        System.out.println(res);
    }
}
