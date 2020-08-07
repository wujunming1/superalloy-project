package python;

import dao.ClusterIntegrationDao;
import dao.DataDao;
import org.python.antlr.ast.Str;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import service.SysAccountService;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.List;
@Service
public class ClusterIntegration {

    @Value("${python.cluster}")
    private String clusterPath;
    @Value("${python.python1}")
    private String pythonPath;
    @Value("${file.data}")
    private String filePath;
    //自动式算法
    public void AutorunCluster(String username,String fileName){
        try{
            System.out.println("-------------");
            System.out.println("start;;;");
            String status ="2";
            String fileAbsolutepath=filePath+username+File.separator+fileName;
//                #python 算法文件路径 算法名称0 K1 数值型混合型标记2 可解释标记3 数据文件名编码4 调用文件路径5 用户名6
            System.out.println(pythonPath+clusterPath+"GA_new.py"+username+fileAbsolutepath);

            String[] args=new String[]{
                    pythonPath,
                    clusterPath+"GA_new.py",
                    username,
                    fileAbsolutepath,
                    status


            };
            Process process=Runtime.getRuntime().exec(args);
            BufferedReader in = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );
            String line;
            while ((line=in.readLine())!=null){
                System.out.println(line);
            }
            in.close();
            process.waitFor();
            System.out.println("end;;;");
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public void AutoclusterInsert(String username,String fileName){
        try{
            System.out.println("-------------");
            System.out.println("start;;;");
            String status ="1";
            String fileAbsolutepath=filePath+username+File.separator+fileName;
//                #python 算法文件路径 算法名称0 K1 数值型混合型标记2 可解释标记3 数据文件名编码4 调用文件路径5 用户名6
            System.out.println(pythonPath+clusterPath+"GA_new.py"+username+fileAbsolutepath);

            String[] args=new String[]{
                    pythonPath,
                    clusterPath+"GA_new.py",
                    username,
                    fileAbsolutepath,
                    status

            };
            Process process=Runtime.getRuntime().exec(args);
            BufferedReader in = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );
            String line;
            while ((line=in.readLine())!=null){
                System.out.println(line);
            }
            in.close();
            process.waitFor();
            System.out.println("end;;;");
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public void AutoclusterDelete(String username,String fileName){
        try{
            System.out.println("-------------");
            System.out.println("start;;;");
            String status ="3";
            String fileAbsolutepath=filePath+username+File.separator+fileName;
//                #python 算法文件路径 算法名称0 K1 数值型混合型标记2 可解释标记3 数据文件名编码4 调用文件路径5 用户名6
//            System.out.println(pythonPath+clusterPath+"GA_new.py"+username+fileAbsolutepath);

            String[] args=new String[]{
                    pythonPath,
                    clusterPath+"GA_new.py",
                    username,
                    fileAbsolutepath,
                    status

            };
            Process process=Runtime.getRuntime().exec(args);
            BufferedReader in = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );
            String line;
            while ((line=in.readLine())!=null){
                System.out.println(line);
            }
            in.close();
            process.waitFor();
            System.out.println("end;;;");
        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
