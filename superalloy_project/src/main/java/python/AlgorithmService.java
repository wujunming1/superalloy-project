package python;

import entity.AlgorithmEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import service.SysAccountService;

import java.io.BufferedReader;
import java.io.InputStreamReader;

@Service
public class AlgorithmService {
    @Autowired
    private SysAccountService sysAccountService;
    @Value("${python.python}")
    private String pythonPath;
    @Value("${python.algorithm}")
    private String algorithmPath;
    @Value("${file.data}")
    private String filePath;

    public String runAlgorithm(String filePath, AlgorithmEntity algo, String picPath){
        /**
         *@Description: 机器学习算法调用
         *@Param: [filePath, algo, picPath]
         *@Return: java.lang.String
         */
        try {
            String username=sysAccountService.getLoginAccount().getUsername();
            System.out.println("-------------");
            System.out.println("start;;;");
            String[] args = new String[] { pythonPath,
                    algorithmPath+algo.getAlgoLocation(),
                    filePath,
                    picPath
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
            return picPath+algo.getAlgoName()+".jpg";
        } catch (Exception e) {
            e.printStackTrace();
            return "";
        }
    }
}
