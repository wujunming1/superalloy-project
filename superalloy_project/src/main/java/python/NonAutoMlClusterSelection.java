package python;

import dao.DataDao;
import dao.NonAutoMlClusterSelectionDao;
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
public class NonAutoMlClusterSelection {
    @Autowired
    private DataDao dataDao;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private NonAutoMlClusterSelectionDao nonAutoMlClusterSelectionDao;
    @Value("${python.nonautomlclusterselection}")
    private String nonautomlclusterselection;
    @Value("${python.python1}")
    private String python3Path;
    @Value("${file.data}")
    private String filePath;

    //执行DBSCAN算法
    public void DbscanModel_Run(String username ,String filename,
                                String Cluster_ALG, Float dbscan_min_samples,
                                Float dbscan_eps, String dbscan_metric){
        try {
            System.out.println("-------------");
            System.out.println("start;;;DBSCAN");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
//    #python .py file_name0 file_path1 dbscan2 eps3 min_samples4 metric 5 username6
            System.out.println(python3Path+nonautomlclusterselection+"Dbscan.py"+filename+
                    fileAbsolutepath+Cluster_ALG+Float.toString(dbscan_eps)+Float.toString(dbscan_min_samples)+dbscan_metric+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"Dbscan.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Float.toString(dbscan_eps),
                    Float.toString(dbscan_min_samples),
                    dbscan_metric,
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

    //执行Kmeans算法
    public void KmeansModel_Run(String username ,String filename,
                                String Cluster_ALG, Float kmeans_min_samples){
        try {
            System.out.println("-------------");
            System.out.println("start;;;KMEANS");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
//    #python .py file_name0 file_path1 Kmeans2 min_samples3 username4
            System.out.println(python3Path+nonautomlclusterselection+"Kmeans.py"+filename+
                    fileAbsolutepath+Cluster_ALG+Float.toString(kmeans_min_samples)+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"Kmeans.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Float.toString(kmeans_min_samples),
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

    //执行Meanshift算法
    public void MeanshiftModel_Run(String username ,String filename,
                                String Cluster_ALG, Float meanshift_bandwidth,
                                Float meanshift_min_freq){
        try {
            System.out.println("-------------");
            System.out.println("start;;;Meanshift");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
//    #python .py file_name0 file_path1 Meanshift eps3 min_samples4 metric 5 username6
            System.out.println(python3Path+nonautomlclusterselection+"Meanshift.py"+filename+
                    fileAbsolutepath+Cluster_ALG+Float.toString(meanshift_bandwidth)+Float.toString(meanshift_min_freq)+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"Meanshift.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Float.toString(meanshift_bandwidth),
                    Float.toString(meanshift_min_freq),
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

    //执行Agglomerative算法
    public void AgglomerativeModel_Run(String username ,String filename,
                                String Cluster_ALG, Float agglomerative_n_clusters,
                                String agglomerative_linkage, String agglomerative_affinity){
        try {
            System.out.println("-------------");
            System.out.println("start;;;Agglomerative");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
//    #python .py file_name0 file_path1 Agglomerative2 eps3 min_samples4 metric 5 username6
            System.out.println(python3Path+nonautomlclusterselection+"Agglomerative.py"+filename+
                    fileAbsolutepath+Cluster_ALG+Float.toString(agglomerative_n_clusters)+agglomerative_linkage+agglomerative_affinity+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"Agglomerative.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Float.toString(agglomerative_n_clusters),
                    agglomerative_linkage,
                    agglomerative_affinity,
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

    //执行Birch算法
    public void BirchModel_Run(String username ,String filename,
                                String Cluster_ALG, Float birch_threshold,
                                Float brich_branching_factor){
        try {
            System.out.println("-------------");
            System.out.println("start;;;Birch");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
//    #python .py file_name0 file_path1 Birch2 eps3 min_samples4 metric 5 username6
            System.out.println(python3Path+nonautomlclusterselection+"Birch.py"+filename+
                    fileAbsolutepath+Cluster_ALG+Float.toString(birch_threshold)+Float.toString(brich_branching_factor)+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"Birch.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Float.toString(birch_threshold),
                    Float.toString(brich_branching_factor),
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

    //执行Affinity算法
    public void AffinityModel_Run(String username ,String filename,
                                String Cluster_ALG, Float affinity_damping){
        try {
            System.out.println("-------------");
            System.out.println("start;;;Affinity");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
//    #python .py file_name0 file_path1 Affinity2 eps3 min_samples4 metric 5 username6
            System.out.println(python3Path+nonautomlclusterselection+"Affinity.py"+filename+
                    fileAbsolutepath+Cluster_ALG+Float.toString(affinity_damping)+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"Affinity.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Float.toString(affinity_damping),
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
    //执行Linear算法
    public void LinearModel_Run(String username ,String filename,
                                String Cluster_ALG){
        try {
            System.out.println("-------------");
            System.out.println("start;;;Linear");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+nonautomlclusterselection+"Linear.py"+filename+
                    fileAbsolutepath+Cluster_ALG+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"Linear.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
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

    //执行rDecisionTree回归算法
    public void rDecisionTreeModel_Run(String username ,String filename,
                                       String Cluster_ALG, int rdecisiontree_depth){
        try {
            System.out.println("-------------");
            System.out.println("start;;;rDecisionTree");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+nonautomlclusterselection+"rDecisionTree.py"+filename+
                    fileAbsolutepath+Cluster_ALG+Integer.toString(rdecisiontree_depth)+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"rDecisionTree.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Integer.toString(rdecisiontree_depth),
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
    //执行SVM回归算法
    public void SVRModel_Run(String username ,String filename,
                             String Cluster_ALG,Float svr_c,String svr_kernel){
        try {
            System.out.println("-------------");
            System.out.println("start;;;SVR");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+nonautomlclusterselection+"SVR.py"+filename+
                    fileAbsolutepath+Cluster_ALG+Float.toString(svr_c)+svr_kernel+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"SVR.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Float.toString(svr_c),
                    svr_kernel,
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
    //KNN回归
    public void rKNNModel_Run(String username ,String filename,
                              String Cluster_ALG,int rknn_k,String rknn_weight){
        try {
            System.out.println("-------------");
            System.out.println("start;;;rKNN");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+nonautomlclusterselection+"rKNN.py"+filename+
                    fileAbsolutepath+Cluster_ALG+Integer.toString(rknn_k)+rknn_weight+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"rKNN.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Integer.toString(rknn_k),
                    rknn_weight,
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
    //执行Svm算法
    public void SvmModel_Run(String username ,String filename,
                             String Cluster_ALG, Float svm_c,Float svm_gamma, Integer svm_degree, String svm_kernel){
        try {
            System.out.println("-------------");
            System.out.println("start;;;Svm");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+nonautomlclusterselection+"Svm.py"+filename+
                    fileAbsolutepath+Cluster_ALG+Float.toString(svm_c)+Float.toString(svm_gamma)
                    +Integer.toString(svm_degree)+svm_kernel+username);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"Svm.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Float.toString(svm_c),
                    Float.toString(svm_gamma),
                    Integer.toString(svm_degree),
                    svm_kernel,
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

    public void KnnModel_Run(String username ,String filename,
                             String Cluster_ALG, Integer n_neighbors, String weights, Integer p){
        try {
            System.out.println("-------------");
            System.out.println("start;;;Knn");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+nonautomlclusterselection+"Knn.py"+filename+
                    fileAbsolutepath+Cluster_ALG +Integer.toString(n_neighbors)+weights+Integer.toString(p));

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"Knn.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    Integer.toString(n_neighbors),
                    weights,
                    Integer.toString(p)
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
    public void FitModel_Run(String username ,String filename, String Cluster_ALG, String ALG_model){
        try {
            System.out.println("-------------");
            System.out.println("start;;;Predict");
            String fileAbsolutepath = filePath+username+ File.separator+filename;
            System.out.println(python3Path+nonautomlclusterselection+"Predict.py"+filename+
                    fileAbsolutepath + Cluster_ALG + ALG_model);

            String[] args = new String[]{
                    python3Path,
                    nonautomlclusterselection+"Predict.py",
                    filename,
                    fileAbsolutepath,
                    Cluster_ALG,
                    ALG_model,
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
