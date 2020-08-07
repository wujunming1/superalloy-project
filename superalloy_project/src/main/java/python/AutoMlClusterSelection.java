package python;

import dao.AutoMlClusterSelectionDao;
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
public class AutoMlClusterSelection {
    @Autowired
    private DataDao dataDao;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private AutoMlClusterSelectionDao autoMlClusterSelectionDao;
    @Value("${python.automlclusterselection}")
    private String automlclusterselectionPath;
    @Value("${python.python1}")
    private String python3Path;
    @Value("${file.data}")
    private String filePath;

    //执行计算元特征的功能
    public void MetaFeature_Run(String username,String filename){
        try {
            System.out.println("-------------");
            System.out.println("start;;;");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            // python .py filename filepath username
            System.out.println(python3Path+automlclusterselectionPath+"Database_meta_features.py"+filename+
                    fileAbsolutepath+username);

            String[] args = new String[]{
                    python3Path,
                    automlclusterselectionPath+"Database_meta_features.py",
                    filename,
                    fileAbsolutepath,
                    username
            };
            Process process = Runtime.getRuntime().exec(args);
            BufferedReader in = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );
            String line;
            while ((line = in.readLine())!=null){
                System.out.println(line);
            }
            in.close();
            process.waitFor();
            System.out.println("end;;;");
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    //执行基于用户、物品、模型的运算
    public void Recommendation_UIM_Run(String username, String filename){
        try {
            System.out.println("-------------");
            System.out.println("start;;;");
//            String fileAbsolutepath = filePath+username+ File.separator+filename;
            // python .py filename username
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+automlclusterselectionPath+"Database_Recommend.py"+filename+
                    fileAbsolutepath + username);

            String[] args = new String[]{
                    python3Path,
                    automlclusterselectionPath+"Database_Recommend.py",
                    filename,
                    fileAbsolutepath,
                    username
            };
            Process process = Runtime.getRuntime().exec(args);
            BufferedReader in = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );
            String line;
            while ((line = in.readLine())!=null){
                System.out.println(line);
            }
            in.close();
            process.waitFor();
            System.out.println("end;;;");
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    //执行基于RF的运算
    public void Recommendation_RF_RUN(String username, String filename){
        try {
            System.out.println("-------------");
            System.out.println("start;;;");
//            String fileAbsolutepath = filePath+username+ File.separator+filename;
            // python .py filename username
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+automlclusterselectionPath+"Database_RF.py"+filename+
                    fileAbsolutepath + username);

            String[] args = new String[]{
                    python3Path,
                    automlclusterselectionPath+"Database_RF.py",
                    filename,
                    fileAbsolutepath,
                    username
            };
            Process process = Runtime.getRuntime().exec(args);
            BufferedReader in = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );
            String line;
            while ((line = in.readLine())!=null){
                System.out.println(line);
            }
            in.close();
            process.waitFor();
            System.out.println("end;;;");
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    //执行贝叶斯优化聚类算法操作
    public void ClusterModel_Run(String username, String filename, String alg){
        try {
            System.out.println("-------------");
            System.out.println("start;;;");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            // python .py filename filepath alg username
            System.out.println(python3Path+automlclusterselectionPath+"Database_Auto_Cluster.py"+filename
                    +fileAbsolutepath + alg + username);

            String[] args = new String[]{
                    python3Path,
                    automlclusterselectionPath+"Database_Auto_Cluster.py",
                    filename,
                    fileAbsolutepath,
                    alg,
                    username
            };
            Process process = Runtime.getRuntime().exec(args);
            BufferedReader in = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );
            String line;
            while ((line = in.readLine())!=null){
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
