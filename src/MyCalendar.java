import java.util.*;

class MyCalendar {

    List<Integer> starts;
    List<Integer> ends;

    public MyCalendar() {
        starts = new ArrayList<>();
        ends = new ArrayList<>();
    }

    public boolean book(int start, int end) {
        boolean firstOverlap = false, secondOverlap = false;
        List<Integer> overlapStarts = new ArrayList<>();
        List<Integer> overlapEnds = new ArrayList<>();
        for(int i = 0; i < starts.size(); i++) {
            int lo = starts.get(i), hi = ends.get(i);
            if(end <= lo || start >= hi) {
                continue;
            } else {
                firstOverlap = true;
                int s = Math.max(lo, start);
                int e = Math.min(hi, end);
                for(int j = 0; j < overlapStarts.size(); j++) {
                    int loo = overlapStarts.get(j);
                    int hii = overlapEnds.get(j);
                    if(e <= loo || s >= hii) {
                        continue;
                    } else {
                        secondOverlap = true;
                        break;
                    }
                }
                if(!secondOverlap) {
                    overlapStarts.add(s);
                    overlapEnds.add(e);
                }
            }
        }
        if(firstOverlap && secondOverlap) {
            return false;
        } else {
            starts.add(start);
            ends.add(end);
            return true;
        }
    }

    public static void main(String[] args) {
        MyCalendar myCalendar = new MyCalendar();
        System.out.println(myCalendar.book(10, 20));
        System.out.println(myCalendar.book(50, 60));
        System.out.println(myCalendar.book(10, 40));
        System.out.println(myCalendar.book(5, 15));
        System.out.println(myCalendar.book(5, 10));
        System.out.println(myCalendar.book(25, 55));
    }
}