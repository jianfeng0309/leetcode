import java.util.*;

/**
 * Created by GuoJianFeng on 11/15/17.
 */
public class TaskSchedule {

    // setting: cannot reorder task, output the whole time
    private int taskSchedule(int[] tasks, int cd) {
        if(tasks == null || tasks.length == 0) return 0;
        // store the end of cd
        Map<Integer, Integer> taskToCooldown = new HashMap<>();

        int slot = 0;
        for(int i = 0; i < tasks.length; i++) {
            int task = tasks[i];
            // if cool down constraint exist
            if(taskToCooldown.containsKey(task) && taskToCooldown.get(task) > slot) {
                slot = taskToCooldown.get(task);
            }
            taskToCooldown.put(task, slot + 1 + cd);
            slot++;
        }
        return slot;
    }

    private String taskScheduleOutput(int[] tasks, int cd) {
        if(tasks == null || tasks.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        Map<Integer, Integer> taskToCD = new HashMap<>();
        int slot = 0;
        for(int i = 0; i < tasks.length; i++) {
            int task = tasks[i];
            int wait = 0;
            if(taskToCD.containsKey(task) && taskToCD.get(task) > slot) {
                wait = taskToCD.get(task) - slot;
                slot = taskToCD.get(task);
            }
            while(wait-- > 0) {
                sb.append("*");
            }
            sb.append(task);
            taskToCD.put(task, slot + 1 + cd);
            slot++;
        }
        return sb.toString();
    }

    private int taskScheduleReorder(int[] tasks, int cd) {
        Map<Integer, Integer> taskToFre = new HashMap<>();
        for(int task : tasks) {
            taskToFre.put(task, taskToFre.getOrDefault(task, 0) + 1);
        }
        int maxFre = 1;
        int times = 1;
        for(int freq : taskToFre.values()) {
            if(freq > maxFre) {
                maxFre = freq;
                times = 1;
            } else if(freq == maxFre) {
                times++;
            }
        }
        return Math.max(tasks.length, (cd + 1) * (maxFre - 1) + times);
    }

    private int taskScheduleLotsTaskSmallCd(int[] tasks, int cd) {
        Queue<Integer> queue = new LinkedList<>();
        Map<Integer, Integer> map = new HashMap<>();

        int slots = 0;
        for(int i = 0; i < tasks.length; i++) {
            int task = tasks[i];
            if(map.containsKey(task) && map.get(task) > slots) {
                slots = map.get(task);
            }
            if(queue.size() == cd + 1) {
                map.remove(queue.poll());
            }
            map.put(task, slots + 1 + cd);
            queue.offer(task);
            slots++;
        }
        return slots;
    }

    public static void main(String[] args) {
        TaskSchedule taskSchedule = new TaskSchedule();
        int[] tasks = new int[]{1,2,1,2,2,2,3};
        System.out.println("Without reordered: " + taskSchedule.taskSchedule(tasks, 4));
        System.out.println("output: " + taskSchedule.taskScheduleOutput(tasks, 4));
    }

}
