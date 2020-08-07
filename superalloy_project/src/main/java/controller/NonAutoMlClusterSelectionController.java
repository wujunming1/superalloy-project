package controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import python.AutoMlClusterSelection;
import service.AutoMlClusterSelectionService;
import service.DataService;
import service.NonAutoMlClusterSelectionService;

import javax.servlet.http.HttpServletRequest;

@Controller
@RequestMapping("/NonAutoMlClusterSelection")
public class NonAutoMlClusterSelectionController {
    @Autowired
    private DataService dataService;
    @Autowired
    private NonAutoMlClusterSelectionService NonAutoMlClusterSelectionService;
    @Autowired
    private AutoMlClusterSelection autoMlClusterSelection;

    //计算DBSCAN
    @RequestMapping("/Dbscan")
    public @ResponseBody
//    eCode:eCode,
//    Cluster_ALG:Cluster_ALG,
//    dbscan_min_samples:dbscan_min_samples,
//    dbscan_eps:dbscan_eps,
//    dbscan_metric:dbscan_metric
    String Dbscan(HttpServletRequest request, Model model,
                        @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
                        @RequestParam("dbscan_min_samples") Float dbscan_min_samples,@RequestParam("dbscan_eps") Float dbscan_eps,
                        @RequestParam("dbscan_metric") String dbscan_metric){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.Dbscan(filename,Cluster_ALG,dbscan_min_samples,dbscan_eps,dbscan_metric);
        System.out.println(json);
        return json;
    }

    //计算Kmeans
    @RequestMapping("/Kmeans")
    public @ResponseBody
//    eCode:eCode,
//    Cluster_ALG:Cluster_ALG,
//    kmeans_min_samples:kmeans_min_samples
    String Kmeans(HttpServletRequest request, Model model,
                  @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
                  @RequestParam("kmeans_min_samples") Float kmeans_min_samples){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.Kmeans(filename,Cluster_ALG,kmeans_min_samples);
        System.out.println(json);
        return json;
    }

    //计算Meanshift
    @RequestMapping("/Meanshift")
    public @ResponseBody
//    eCode:eCode,
//    Cluster_ALG:Cluster_ALG,
//    meanshift_bandwidth:meanshift_bandwidth
//    meanshift_min_freq:meanshift_min_freq
    String Meanshift(HttpServletRequest request, Model model,
                  @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
                  @RequestParam("meanshift_bandwidth") Float meanshift_bandwidth,@RequestParam("meanshift_min_freq") Float meanshift_min_freq){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.Meanshift(filename,Cluster_ALG,meanshift_bandwidth,meanshift_min_freq);
        System.out.println(json);
        return json;
    }

    //计算Agglomerative
    @RequestMapping("/Agglomerative")
    public @ResponseBody
//    eCode:eCode,
//    Cluster_ALG:Cluster_ALG,
//    dbscan_min_samples:dbscan_min_samples,
//    dbscan_eps:dbscan_eps,
//    dbscan_metric:dbscan_metric
    String Agglomerative(HttpServletRequest request, Model model,
                  @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
                  @RequestParam("agglomerative_n_clusters") Float agglomerative_n_clusters,@RequestParam("agglomerative_linkage") String agglomerative_linkage,
                  @RequestParam("agglomerative_affinity") String agglomerative_affinity){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.Agglomerative(filename,Cluster_ALG,agglomerative_n_clusters,agglomerative_linkage,agglomerative_affinity);
        System.out.println(json);
        return json;
    }

    //计算Birch
    @RequestMapping("/Birch")
    public @ResponseBody
    String Birch(HttpServletRequest request, Model model,
                  @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
                  @RequestParam("birch_threshold") Float birch_threshold, @RequestParam("brich_branching_factor") Float brich_branching_factor){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.Birch(filename,Cluster_ALG,birch_threshold,brich_branching_factor);
        System.out.println(json);
        return json;
    }

    //计算Affinity
    @RequestMapping("/Affinity")
    public @ResponseBody
//    eCode:eCode,
//    Cluster_ALG:Cluster_ALG,
//    dbscan_min_samples:dbscan_min_samples,
//    dbscan_eps:dbscan_eps,
//    dbscan_metric:dbscan_metric
    String Affinity(HttpServletRequest request, Model model,
                  @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
                  @RequestParam("affinity_damping") Float affinity_damping){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.Affinity(filename,Cluster_ALG,affinity_damping);
        System.out.println(json);
        return json;
    }
    //计算Linear回归
    @RequestMapping("/Linear")
    public @ResponseBody
    /**
     *@Param:eCode:eCode,
     *@Param:Cluster_ALG:Cluster_ALG,
     *@Param:dbscan_min_samples:dbscan_min_samples,
     *@Param:dbscan_eps:dbscan_eps,
     *@Param:dbscan_metric:dbscan_metric
     */
    String Linear(HttpServletRequest request, Model model,
                  @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.Linear(filename,Cluster_ALG);
        System.out.println(json);
        return json;
    }

    //计算DecisionTree回归
    @RequestMapping("/rDecisionTree")
    public @ResponseBody
//    eCode:eCode,
//    Cluster_ALG:Cluster_ALG,
//    dbscan_min_samples:dbscan_min_samples,
//    dbscan_eps:dbscan_eps,
//    dbscan_metric:dbscan_metric
    String DecisionTree(HttpServletRequest request, Model model,
                        @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
                        @RequestParam("rdecisiontree_depth") int rdecisiontree_depth){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.rDecisionTree(filename,Cluster_ALG,rdecisiontree_depth);
        System.out.println(json);
        return json;
    }
    //计算SVM回归
    @RequestMapping("/Svr")
    public @ResponseBody
//    eCode:eCode,
//    Cluster_ALG:Cluster_ALG,
//    dbscan_min_samples:dbscan_min_samples,
//    dbscan_eps:dbscan_eps,
//    dbscan_metric:dbscan_metric
    String SVM(HttpServletRequest request, Model model,
               @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
               @RequestParam("svr_c") Float svr_c, @RequestParam("svr_kernel") String svr_kernel){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.SVR(filename,Cluster_ALG,svr_c,svr_kernel);
        System.out.println(json);
        return json;
    }
    //计算KNN回归
    @RequestMapping("/rKnn")
    public @ResponseBody
//    eCode:eCode,
//    Cluster_ALG:Cluster_ALG,
//    dbscan_min_samples:dbscan_min_samples,
//    dbscan_eps:dbscan_eps,
//    dbscan_metric:dbscan_metric
    String KNN(HttpServletRequest request, Model model,
               @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
               @RequestParam("rknn_k") int rknn_k, @RequestParam("rknn_weight") String rknn_weight){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.rKNN(filename,Cluster_ALG,rknn_k,rknn_weight);
        System.out.println(json);
        return json;
    }

    //计算Svm
    @RequestMapping("/Svm")
    public @ResponseBody
//    eCode:eCode,
//    Cluster_ALG:Cluster_ALG,
//    dbscan_min_samples:dbscan_min_samples,
//    dbscan_eps:dbscan_eps,
//    dbscan_metric:dbscan_metric
    String Svm(HttpServletRequest request, Model model,
               @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
               @RequestParam("svm_c") Float svm_c, @RequestParam("svm_gamma") Float svm_gamma,
               @RequestParam("svm_degree") Integer svm_degree,@RequestParam("svm_kernel") String svm_kernel){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.Svm(filename,Cluster_ALG,svm_c, svm_gamma, svm_degree, svm_kernel);
        System.out.println(json);
        return json;
    }

    //计算Knn
    @RequestMapping("/Knn")
    public @ResponseBody
//    eCode:eCode,
//    Cluster_ALG:Cluster_ALG,
//    dbscan_min_samples:dbscan_min_samples,
//    dbscan_eps:dbscan_eps,
//    dbscan_metric:dbscan_metric
    String Knn(HttpServletRequest request, Model model,
               @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
               @RequestParam("n_neighbors") Integer n_neighbors, @RequestParam("weights") String weights,
               @RequestParam("p") Integer p){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.Knn(filename,Cluster_ALG,n_neighbors, weights, p);
        System.out.println(json);
        return json;
    }

    @RequestMapping("/Predict")
    public @ResponseBody
    /**
     *@Description: 对上传数据集预测的接口
     *@Param: eCode：已上传的数据集的唯一标识
     *@Param: ALG_model：已经保存模型的文件名(.m文件)
     *@Return: java.lang.String
     */
    String Predict(HttpServletRequest request, Model model,
                   @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG, @RequestParam("ALG_model") String ALG_model){
        String filename = dataService.getRecordByECode(eCode);
        String json = NonAutoMlClusterSelectionService.FitModel(filename,Cluster_ALG,ALG_model);
        System.out.println(json);
        return json;
    }

}
