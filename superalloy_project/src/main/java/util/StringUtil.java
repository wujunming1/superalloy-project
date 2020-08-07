package util;

/**
 * Created by piki on 2018/4/2.
 */
public class StringUtil {
    public static void main(String[] args) {
        String s = "1";
        System.out.println(s.getClass().toString());
    }
    public static Boolean isEmpty(String s){
        return s==null||s==""||s.length()<=0;
    }
}
