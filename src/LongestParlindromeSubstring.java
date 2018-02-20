import java.util.HashSet;

/**
 * Created by GuoJianFeng on 9/30/17.
 */
public class LongestParlindromeSubstring {
    int start = 0;
    int maxLen = 0;

    private void extend(String str, int from, int to) {
        while (from >= 0 && to < str.length() && str.charAt(from) == str.charAt(to)) {
            from--;
            to++;
        }
        if (to - from - 1 > maxLen) {
            maxLen = to - from - 1;
            start = from + 1;
        }
    }

    public String longestPalindrome(String s) {
        if (s == null) return null;
        int len = s.length();
        if (len < 2) {
            return s;
        }

        for (int i = 0; i < len - 1; i++) {
            extend(s, i, i);
            extend(s, i, i + 1);
        }
        return s.substring(start, start + maxLen);
    }

}
