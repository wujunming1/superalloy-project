package python;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * Created by piki on 2018/3/9.
 */
@Service
public class ReadCifService {

    @Value("${python.python}")
    private String pythonPath;
    @Value("${python.rcif}")
    private String rcifPath;
    public ReadCifService() {
    }

    public void readCif(String serverFile){
        
        
        try {
            System.out.println("-------------");
            System.out.println("start;;;");
            // 特征分析的文件(excel或csv格式的文件)的绝对路径
            String[] args = new String[] { pythonPath,
                    rcifPath,
                    serverFile
            };
            Process pr = Runtime.getRuntime().exec(args);
            BufferedReader in = new BufferedReader(new InputStreamReader(pr.getInputStream()));
            String line;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            pr.waitFor();
            System.out.println("end");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
