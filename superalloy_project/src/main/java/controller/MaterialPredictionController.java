package controller;

import com.alibaba.fastjson.JSONObject;
import dao.DataDao;
import entity.SysAccount;
import javafx.beans.binding.ObjectExpression;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import python.AutoMlClusterSelection;
import service.*;

import javax.servlet.http.HttpServletRequest;
import java.io.File;
import java.util.*;

@Controller
@RequestMapping("/MaterialPrediction")
public class MaterialPredictionController{
    @Value("${file.data}")
    private String filePath;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private ECodeService eCodeService;

    @Autowired
    private DataService dataService;
    @Autowired
    private ExcelAnalysisService excelAnalysisService;
    @Autowired
    private ReadResultService readResultService;
    @Autowired
    private CsvAnalysisService csvAnalysisService;
    @Autowired
    private CreepPredictionService creepPredictionService;
//    @Autowired
//    private AutoMlClusterSelection autoMlClusterSelection;
    @Autowired
    private DataDao dataDao;

    //计算DBSCAN
    @RequestMapping("/ClusterDivision")
    public @ResponseBody
    String Clusterdivision(HttpServletRequest request, Model model, @RequestParam("eCode") String eCode,
                           @RequestParam("clusterNum") String clusterNum) {
        String filename = dataService.getRecordByECode(eCode);
        String status = creepPredictionService.ClusterAlloy(clusterNum, filename);
        String username = sysAccountService.getLoginAccount().getUsername();
        System.out.println(status);
        String pathname1 = filePath+username+"\\cluster_split_data_"+username+".xlsx";
        ArrayList<ArrayList<Object>> arrayLists1 = excelAnalysisService.readExcel(pathname1);
        ArrayList<Object> headerTitle1 = arrayLists1.get(0);
        List<Map<Object, Object>> maplist1 = new ArrayList<Map<Object, Object>>();
        for(int i=1;i<arrayLists1.size();i++){
            Map<Object, Object> map1 = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle1.size();j++){
                map1.put(headerTitle1.get(j),arrayLists1.get(i).get(j));
            }
            maplist1.add(map1);
        }
        System.out.println("222dadad"+maplist1);
        System.out.println(JSONObject.toJSONString(maplist1));
        System.out.println("sdddd"+arrayLists1);
        ArrayList<ArrayList<Object>>  returnData1 = null;
        returnData1 = dataDao.listToDB(arrayLists1,pathname1,"clusterResult",eCode);//解析excel后写入mongodb//csv
        String pathname2 = filePath+username+"\\clusterVisualation_"+username+".xlsx";
        ArrayList<ArrayList<Object>> arrayLists2 = excelAnalysisService.readExcel(pathname2);
        ArrayList<Object> headerTitle2 = arrayLists2.get(0);
        List<Map<Object, Object>> maplist2 = new ArrayList<Map<Object, Object>>();
        for(int i=1;i<arrayLists2.size();i++){
            Map<Object, Object> map2 = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle2.size();j++){
                map2.put(headerTitle2.get(j),arrayLists2.get(i).get(j));
            }
            maplist2.add(map2);
        }
        System.out.println("hello");
        System.out.println("222dadad"+maplist2);
        System.out.println(JSONObject.toJSONString(maplist2));
        System.out.println("sdddd"+arrayLists2);
        ArrayList<ArrayList<Object>>  returnData2=null;
        returnData2 = dataDao.listToDB(arrayLists2,pathname2,"clusterResult",eCode);//解析excel后写入mongodb//csv
        System.out.println("---------------------------------------------");
        String pathname3 = filePath+username+"\\cluster_sample_"+username+".xlsx";
        ArrayList<ArrayList<Object>> arrayLists3 = excelAnalysisService.readExcel(pathname3);
        ArrayList<Object> headerTitle3 = arrayLists3.get(0);
        List<Map<Object, Object>> maplist3 = new ArrayList<Map<Object, Object>>();
        for(int i=1;i<arrayLists3.size();i++){
            Map<Object, Object> map3 = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle3.size();j++){
                map3.put(headerTitle3.get(j),arrayLists3.get(i).get(j));
            }
            maplist3.add(map3);
        }
        System.out.println("222dadad"+maplist3);
        System.out.println(JSONObject.toJSONString(maplist3));
        System.out.println("sdddd"+arrayLists3);
        ArrayList<ArrayList<Object>>  returnData3=null;
        returnData3=dataDao.listToDB(arrayLists3,pathname3,"clusterResult",eCode);//解析excel后写入mongodb//csv
        Map<Object, Object> map = new LinkedHashMap<Object, Object>();

        map.put("headTitle1", JSONObject.toJSONString(headerTitle1));
        map.put("clusterDivision", JSONObject.toJSONString(maplist1));
        map.put("clusterVisual", JSONObject.toJSONString(maplist2));
        map.put("clusterSample", JSONObject.toJSONString(maplist3));
        return JSONObject.toJSONString(map);

    }
    @RequestMapping("/Cluster_Re_Analysis")
    public @ResponseBody
    String Cluster_Re_Analysis(HttpServletRequest request, Model model, @RequestParam("eCode") String eCode) {
        String username = sysAccountService.getLoginAccount().getUsername();
        String pathname1 = filePath+username+"\\cluster_Re_"+username+".xlsx";
        ArrayList<ArrayList<Object>> arrayLists2 = excelAnalysisService.readExcel(pathname1);
        ArrayList<Object> headerTitle2 = arrayLists2.get(0);
        List<Map<Object, Object>> maplist = new ArrayList<Map<Object, Object>>();
        for(int i=1;i<arrayLists2.size();i++){
            Map<Object, Object> map1 = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle2.size();j++){
                map1.put(headerTitle2.get(j),arrayLists2.get(i).get(j));
            }
            maplist.add(map1);
        }
        System.out.println("hello");
        System.out.println("222dadad"+maplist);
        System.out.println(JSONObject.toJSONString(maplist));
        System.out.println("sdddd"+arrayLists2);
        ArrayList<ArrayList<Object>>  returnData=null;
        returnData=dataDao.listToDB(arrayLists2,pathname1,"clusterResult",eCode);//解析excel后写入mongodb//csv
//        JSONObject jsonObject = new JSONObject();
//        jsonObject.put("headerTitle", JSONObject.toJSONString(headerTitle));
        return JSONObject.toJSONString(maplist);
    }
    @RequestMapping("/Cluster_T_Analysis")
    public @ResponseBody
    String Cluster_T_Analysis(HttpServletRequest request, Model model, @RequestParam("eCode") String eCode) {
        String username = sysAccountService.getLoginAccount().getUsername();
        String pathname1 = filePath+username+"\\cluster_T_"+username+".xlsx";
        ArrayList<ArrayList<Object>> arrayLists2 = excelAnalysisService.readExcel(pathname1);
        ArrayList<Object> headerTitle2 = arrayLists2.get(0);
        List<Map<Object, Object>> maplist = new ArrayList<Map<Object, Object>>();
        for(int i=1;i<arrayLists2.size();i++){
            Map<Object, Object> map1 = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle2.size();j++){
                map1.put(headerTitle2.get(j),arrayLists2.get(i).get(j));
            }
            maplist.add(map1);
        }
        System.out.println("hello");
        System.out.println("222dadad"+maplist);
        System.out.println(JSONObject.toJSONString(maplist));
        System.out.println("sdddd"+arrayLists2);
        ArrayList<ArrayList<Object>>  returnData=null;
        returnData=dataDao.listToDB(arrayLists2,pathname1,"clusterResult",eCode);//解析excel后写入mongodb//csv
//        JSONObject jsonObject = new JSONObject();
//        jsonObject.put("headerTitle", JSONObject.toJSONString(headerTitle));
        return JSONObject.toJSONString(maplist);
    }
    @RequestMapping("/Cluster_S_Analysis")
    public @ResponseBody
    String Cluster_S_Analysis(HttpServletRequest request, Model model, @RequestParam("eCode") String eCode) {
        String username = sysAccountService.getLoginAccount().getUsername();
        String pathname1 = filePath+username+"\\cluster_S_"+username+".xlsx";
        ArrayList<ArrayList<Object>> arrayLists2 = excelAnalysisService.readExcel(pathname1);
        ArrayList<Object> headerTitle2 = arrayLists2.get(0);
        List<Map<Object, Object>> maplist = new ArrayList<Map<Object, Object>>();
        for(int i=1;i<arrayLists2.size();i++){
            Map<Object, Object> map1 = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle2.size();j++){
                map1.put(headerTitle2.get(j),arrayLists2.get(i).get(j));
            }
            maplist.add(map1);
        }
        System.out.println("hello");
        System.out.println("222dadad"+maplist);
        System.out.println(JSONObject.toJSONString(maplist));
        System.out.println("sdddd"+arrayLists2);
        ArrayList<ArrayList<Object>>  returnData=null;
        returnData=dataDao.listToDB(arrayLists2,pathname1,"clusterResult",eCode);//解析excel后写入mongodb//csv
        return JSONObject.toJSONString(maplist);
    }
    //计算每个簇上模型的适应度比较
    @RequestMapping("/DCSA_analysis")
    public @ResponseBody
    String Fitness(HttpServletRequest request, Model model, @RequestParam("eCode") String eCode,
                   @RequestParam("svr_kernel") String svr_kernel, @RequestParam("svr_c") String svr_c
            , @RequestParam("gpr_alpha") String gpr_alpha, @RequestParam("gpr_optimazed") String gpr_optimazed,
                   @RequestParam("rf_estimators") String rf_estimators,
                   @RequestParam("rf_max_depth") String  rf_max_depth,
                   @RequestParam("lr_alpha") String lr_alpha, @RequestParam("rr_alpha") String rr_alpha ) {
        String username = sysAccountService.getLoginAccount().getUsername();
        File file = new File( filePath+username+"\\cluster_Fitness_"+username+".xlsx");
        if (file.isFile() && file.exists()) {
            file.delete();
        }
        File file1 = new File( filePath+username+"\\cluster_Fitness_"+username+".xlsx");
        if (file1.isFile() && file1.exists()) {
            file1.delete();
        }
        File file2 = new File( filePath+username+"\\cluster_TruePred_"+username+".xlsx");
        if (file2.isFile() && file2.exists()) {
            file2.delete();
        }
        String filename = dataService.getRecordByECode(eCode);
        String status = creepPredictionService.DCSA_analysis(filename, svr_kernel, svr_c,
                gpr_alpha, gpr_optimazed,
                rf_estimators, rf_max_depth, lr_alpha, rr_alpha);

        System.out.println(status);
        String pathname1 = filePath+username+"\\cluster_Fitness_"+username+".xlsx";
        ArrayList<ArrayList<Object>> arrayLists1 = excelAnalysisService.readExcel(pathname1);
        ArrayList<Object> headerTitle1 = arrayLists1.get(0);
        List<Map<Object, Object>> maplist1 = new ArrayList<Map<Object, Object>>();
        for(int i=1;i<arrayLists1.size();i++){
            Map<Object, Object> map1 = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle1.size();j++){
                map1.put(headerTitle1.get(j),arrayLists1.get(i).get(j));
            }
            maplist1.add(map1);
        }
        System.out.println("222dadad"+maplist1);
        System.out.println(JSONObject.toJSONString(maplist1));
        System.out.println("sdddd"+arrayLists1);
        ArrayList<ArrayList<Object>>  returnData1=null;
        returnData1=dataDao.listToDB(arrayLists1,pathname1,"clusterResult",eCode);//解析excel后写入mongodb//csv
        String pathname2 = filePath+username+"\\cluster_Fitness_"+username+".xlsx";
        ArrayList<ArrayList<Object>> arrayLists2 = excelAnalysisService.readExcel(pathname2);
        ArrayList<Object> headerTitle2 = arrayLists2.get(0);
        List<Map<Object, Object>> maplist2 = new ArrayList<Map<Object, Object>>();
        for(int i=1;i<arrayLists2.size();i++){
            Map<Object, Object> map2 = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle2.size();j++){
                map2.put(headerTitle2.get(j),arrayLists2.get(i).get(j));
            }
            maplist2.add(map2);
        }
        System.out.println("222dadad"+maplist2);
        System.out.println(JSONObject.toJSONString(maplist2));
        System.out.println("sdddd"+arrayLists2);
        ArrayList<ArrayList<Object>>  returnData2=null;
        returnData2=dataDao.listToDB(arrayLists2,pathname2,"clusterResult",eCode);//解析excel后写入mongodb//csv
        String pathname3 = filePath+username+"\\cluster_TruePred_"+username+".xlsx";
        ArrayList<ArrayList<Object>> arrayLists3 = excelAnalysisService.readExcel(pathname3);
        ArrayList<Object> headerTitle3 = arrayLists3.get(0);
        List<Map<Object, Object>> maplist3 = new ArrayList<Map<Object, Object>>();
        for(int i=1;i<arrayLists3.size();i++){
            Map<Object, Object> map3 = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle3.size();j++){
                map3.put(headerTitle3.get(j),arrayLists3.get(i).get(j));
            }
            maplist3.add(map3);
        }
        System.out.println("222dadad"+maplist3);
        System.out.println(JSONObject.toJSONString(maplist3));
        System.out.println("sdddd"+arrayLists3);
        ArrayList<ArrayList<Object>>  returnData3=null;
        returnData3=dataDao.listToDB(arrayLists3,pathname3,"clusterResult",eCode);//解析excel后写入mongodb//csv
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("headerTitle3", JSONObject.toJSONString(headerTitle3));
        jsonObject.put("Fitness", JSONObject.toJSONString(maplist1));
        jsonObject.put("FitnessVisual", JSONObject.toJSONString(maplist2));
        jsonObject.put("PredictedTruePred", JSONObject.toJSONString(maplist3));
        return jsonObject.toJSONString();
    }
    //预测结果：真实值与预测值的比较
    @RequestMapping("/TruePred")
    public @ResponseBody
    String TruePred(HttpServletRequest request, Model model, @RequestParam("eCode") String eCode ) {
        String username = sysAccountService.getLoginAccount().getUsername();
        String pathname1 = filePath+username+"\\cluster_TruePred_"+username+".xlsx";
        ArrayList<ArrayList<Object>> arrayLists = excelAnalysisService.readExcel(pathname1);
        ArrayList<Object> headerTitle = arrayLists.get(0);
        List<Map<Object, Object>> maplist = new ArrayList<Map<Object, Object>>();
        for(int i=1;i<arrayLists.size();i++){
            Map<Object, Object> map = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle.size();j++){
                map.put(headerTitle.get(j),arrayLists.get(i).get(j));
            }
            maplist.add(map);
        }
        System.out.println("222dadad"+maplist);
        System.out.println(JSONObject.toJSONString(maplist));
        System.out.println("sdddd"+arrayLists);
        ArrayList<ArrayList<Object>>  returnData=null;
        returnData=dataDao.listToDB(arrayLists,pathname1,"clusterResult",eCode);//解析excel后写入mongodb//csv
        return JSONObject.toJSONString(maplist);
    }
    //蠕变性能优化
    @RequestMapping("/CreepOptimazation")
    public @ResponseBody
    String Creepinteration(HttpServletRequest request, Model model, @RequestParam("eCode") String eCode ) {
        String pathname1 = "D:\\superlloy\\data\\piki\\creep_iteration.xlsx";
        ArrayList<ArrayList<Object>> arrayLists = excelAnalysisService.readExcel(pathname1);
        ArrayList<Object> headerTitle = arrayLists.get(0);
        List<Map<Object, Object>> maplist = new ArrayList<Map<Object, Object>>();
        System.out.println("hello");
        for(int i=1;i<arrayLists.size();i++){
            Map<Object, Object> map = new LinkedHashMap<Object, Object>();
            for(int j=0;j<headerTitle.size();j++){
                System.out.println(111);
                map.put(headerTitle.get(j),arrayLists.get(i).get(j));
            }
            maplist.add(map);
        }
        System.out.println(111);
        System.out.println("222dadad"+maplist);
        System.out.println(JSONObject.toJSONString(maplist));
        System.out.println("sdddd"+arrayLists);
        ArrayList<ArrayList<Object>>  returnData=null;
        returnData=dataDao.listToDB(arrayLists,pathname1,"clusterResult",eCode);//解析excel后写入mongodb//csv
        return JSONObject.toJSONString(maplist);
    }
}
