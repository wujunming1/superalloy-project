package service;

import com.alibaba.fastjson.JSONObject;
import dao.DataDao;
import dao.NonAutoMlClusterSelectionDao;
import entity.ClusterModelEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import python.NonAutoMlClusterSelection;

import java.util.List;

@Service
public class NonAutoMlClusterSelectionService {
    @Autowired
    private DataDao dataDao;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private NonAutoMlClusterSelectionDao nonAutoMlClusterSelectionDao;
    @Autowired
    private NonAutoMlClusterSelection nonAutoMlClusterSelection;

    //第一次计算Dbscan聚类算法
    public String Dbscan(String filename,String Cluster_ALG,
                         Float dbscan_min_samples,Float dbscan_eps,
                         String dbscan_metric){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        DbscanModel_Run(username, filename,Cluster_ALG,
                dbscan_min_samples, dbscan_eps, dbscan_metric);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void DbscanModel_Run(String username ,String filename,
                                 String Cluster_ALG, Float dbscan_min_samples,
                                 Float dbscan_eps, String dbscan_metric){
        nonAutoMlClusterSelection.DbscanModel_Run(username, filename,Cluster_ALG,
                dbscan_min_samples, dbscan_eps, dbscan_metric);
    }

    //第一次计算Kmeans聚类算法
    public String Kmeans(String filename,String Cluster_ALG,
                         Float kmeans_min_samples){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        KmeansModel_Run(username, filename, Cluster_ALG, kmeans_min_samples);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void KmeansModel_Run(String username ,String filename,
                                 String Cluster_ALG, Float kmeans_min_samples){
        nonAutoMlClusterSelection.KmeansModel_Run(username, filename, Cluster_ALG, kmeans_min_samples);
    }

    //第一次计算Meanshift聚类算法
    public String Meanshift(String filename,String Cluster_ALG,
                         Float meanshift_bandwidth,Float meanshift_min_freq){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        MeanshiftModel_Run(username, filename,Cluster_ALG,
                meanshift_bandwidth, meanshift_min_freq);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void MeanshiftModel_Run(String username ,String filename,
                                 String Cluster_ALG, Float meanshift_bandwidth,
                                 Float meanshift_min_freq){
        nonAutoMlClusterSelection.MeanshiftModel_Run(username, filename,Cluster_ALG, meanshift_bandwidth, meanshift_min_freq);
    }

    //第一次计算Agglomerative聚类算法
    public String Agglomerative(String filename,String Cluster_ALG,
                         Float agglomerative_n_clusters,String agglomerative_linkage,
                         String agglomerative_affinity){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        AgglomerativeModel_Run(username, filename,Cluster_ALG,
                agglomerative_n_clusters, agglomerative_linkage, agglomerative_affinity);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void AgglomerativeModel_Run(String username ,String filename,
                                 String Cluster_ALG, Float agglomerative_n_clusters,
                                 String agglomerative_linkage, String agglomerative_affinity){
        nonAutoMlClusterSelection.AgglomerativeModel_Run(username, filename,Cluster_ALG,
                agglomerative_n_clusters, agglomerative_linkage, agglomerative_affinity);
    }

    //第一次计算Birch聚类算法
    public String Birch(String filename,String Cluster_ALG,
                         Float birch_threshold,Float brich_branching_factor){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        BirchModel_Run(username, filename,Cluster_ALG,
                birch_threshold, brich_branching_factor);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void BirchModel_Run(String username ,String filename,
                                 String Cluster_ALG, Float birch_threshold,
                                 Float brich_branching_factor){
        nonAutoMlClusterSelection.BirchModel_Run(username, filename,Cluster_ALG,
                birch_threshold, brich_branching_factor);
    }

    //第一次计算Affinity聚类算法
    public String Affinity(String filename,String Cluster_ALG,
                         Float affinity_damping){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        AffinityModel_Run(username, filename,Cluster_ALG,
                affinity_damping);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void AffinityModel_Run(String username ,String filename,
                                 String Cluster_ALG, Float affinity_damping){
        nonAutoMlClusterSelection.AffinityModel_Run(username, filename,Cluster_ALG,
                affinity_damping);
    }
    //第一次计算Linear回归算法
    public String Linear(String filename,String Cluster_ALG){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        LinearModel_Run(username, filename,Cluster_ALG);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void LinearModel_Run(String username ,String filename,
                                 String Cluster_ALG){
        nonAutoMlClusterSelection.LinearModel_Run(username, filename,Cluster_ALG);
    }
    //第一次计算DecisionTree回归算法
    public String rDecisionTree(String filename,String Cluster_ALG,int rdecisiontree_depth){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        rDecisionTreeModel_Run(username, filename,Cluster_ALG,rdecisiontree_depth);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void rDecisionTreeModel_Run(String username ,String filename,
                                        String Cluster_ALG, int rdecisiontree_depth){
        nonAutoMlClusterSelection.rDecisionTreeModel_Run(username, filename,Cluster_ALG,rdecisiontree_depth);
    }
    //第一次计算SVM回归算法
    public String SVR(String filename,String Cluster_ALG,Float svr_c,String svr_kernel){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        SVRModel_Run(username, filename,Cluster_ALG,svr_c,svr_kernel);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void SVRModel_Run(String username ,String filename,
                              String Cluster_ALG,Float svr_c,String svr_kernel){
        nonAutoMlClusterSelection.SVRModel_Run(username, filename,Cluster_ALG,svr_c,svr_kernel);
    }
    //第一次计算KNN回归算法
    public String rKNN(String filename, String Cluster_ALG, int rknn_k, String rknn_weight){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        rKNNModel_Run(username, filename,Cluster_ALG,rknn_k,rknn_weight);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void rKNNModel_Run(String username ,String filename,
                               String Cluster_ALG, int rknn_k, String rknn_weight){
        nonAutoMlClusterSelection.rKNNModel_Run(username, filename,Cluster_ALG,rknn_k,rknn_weight);
    }

    //第一次计算Svm分类算法
    public String Svm(String filename,String Cluster_ALG,
                      Float svm_c,Float svm_gamma, Integer svm_degree, String svm_kernel){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        SvmModel_Run(username, filename,Cluster_ALG,
                svm_c, svm_gamma, svm_degree, svm_kernel);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void SvmModel_Run(String username ,String filename,
                              String Cluster_ALG, Float svm_c,Float svm_gamma,
                              Integer svm_degree, String svm_kernel){
        nonAutoMlClusterSelection.SvmModel_Run(username, filename,Cluster_ALG,
                svm_c, svm_gamma, svm_degree, svm_kernel);
    }

    //第一次计算Knn分类算法
    public String Knn(String filename, String Cluster_ALG,
                      Integer n_neighbors, String weights, Integer p){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelectionDao.AlgModel_Clean(username,filename,Cluster_ALG);
        KnnModel_Run(username, filename,Cluster_ALG, n_neighbors, weights, p);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);

        return JSONObject.toJSONString(result);
    }
    private void KnnModel_Run(String username ,String filename,
                              String Cluster_ALG, Integer n_neighbors, String weights, Integer p){
        nonAutoMlClusterSelection.KnnModel_Run(username, filename,Cluster_ALG, n_neighbors, weights, p);
    }

    //获取所有用户已经有的model
    public String Model(String username){
        List<ClusterModelEntity> result = nonAutoMlClusterSelectionDao.AlgModel_getResultbyUsername(username);
        return JSONObject.toJSONString(result);
    }
    //对于选定的数据集和选定的m模型进行预测
    public String FitModel(String filename, String Cluster_ALG,String ALG_model){
        String username = sysAccountService.getLoginAccount().getUsername();
        nonAutoMlClusterSelection.FitModel_Run(username,filename,Cluster_ALG,ALG_model);
        ClusterModelEntity result = nonAutoMlClusterSelectionDao.AlgModel_getResult(username,filename,Cluster_ALG);
        return JSONObject.toJSONString(result);
    }
}
