package python;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

@Service
public class Preprocessing {

    @Value("${python.preprocessing}")
    private String preprocessing;
    @Value("${python.python1}")
    private String python3Path;
    @Value("${file.data}")
    private String filePath;

    public void Encode_Run(String username ,String filename, String Cluster_ALG,
                           String onehot, String numerical, String numerical_dict){
        try {
            System.out.println("-------------");
            System.out.println("start;;;ENCODING");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+preprocessing+"Encode.py"+filename+
                    fileAbsolutepath+Cluster_ALG+onehot+numerical+numerical_dict+username);

            String[] args = new String[]{
                    python3Path,
                    preprocessing+"Encode.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    onehot,
                    numerical,
                    numerical_dict,
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
    public void Nullcheck_Run(String username ,String filename, String Cluster_ALG,
                           String null_type){
        try {
            System.out.println("-------------");
            System.out.println("start;;;NULLCHECK");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+preprocessing+"Nullcheck.py"+filename+
                    fileAbsolutepath+Cluster_ALG+null_type+username);

            String[] args = new String[]{
                    python3Path,
                    preprocessing+"Nullcheck.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    null_type,
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
    public void Nullvalue_Run(String username ,String filename, String Cluster_ALG,
                              String null_type,String null_method,String null_para){
        try {
            System.out.println("-------------");
            System.out.println("start;;;NULLVALUE");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+preprocessing+"Nullvalue.py"+filename+
                    fileAbsolutepath+Cluster_ALG+null_type+username);

            String[] args = new String[]{
                    python3Path,
                    preprocessing+"Nullvalue.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    null_type,
                    null_method,
                    null_para,
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
