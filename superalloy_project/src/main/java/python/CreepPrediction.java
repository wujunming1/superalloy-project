package python;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

@Service
public class CreepPrediction {

    @Value("${python.creepPrediction}")
    private String cluster_alloy;
    @Value("${python.python1}")
    private String python3Path;
    @Value("${file.data}")
    private String filePath;

    public void ClusterSuperalloy(String clusterNum ,String filename, String username){
        try {
            System.out.println("-------------");
            System.out.println("start;;;Cluster");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            String generated_path = filePath+username+"\\";
            System.out.println("ssssssddddd"+fileAbsolutepath);
            System.out.println("ssssssddddd1"+generated_path);
            System.out.println("ssssssddddd2"+clusterNum);
            System.out.println("ssssssddddd3"+username);
            String[] args = new String[]{
                    python3Path,
                    cluster_alloy+"Cluster_superalloy.py",
                    clusterNum,
                    fileAbsolutepath,
                    generated_path,
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
    public void DCSA_analysis(String filename,
                              String username, String kernel, String C,
                              String gpr_alpha, String optimized_num, String estimators,
                              String maxDepth, String lr_alpha, String rr_alpha){
        try {
            System.out.println("-------------");
            System.out.println("start;;;DCSA_Prediction");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            String generated_path = filePath+username+"\\";
            System.out.println("ssssssddddd"+fileAbsolutepath);
            System.out.println("ssssssddddd1"+generated_path);
//            System.out.println("ssssssddddd2"+clusterNum);
            System.out.println("ssssssddddd3"+username);
//            System.out.println(python3Path+preprocessing+"Encode.py"+filename+
//                    fileAbsolutepath+Cluster_ALG+onehot+numerical+numerical_dict+username);

            String[] args = new String[]{
                    python3Path,
                    cluster_alloy+"DCSA_analysis.py",
                    kernel,
                    C,
                    gpr_alpha,
                    optimized_num,
                    estimators,
                    maxDepth,
                    lr_alpha,
                    rr_alpha,
                    fileAbsolutepath,
                    generated_path,
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
