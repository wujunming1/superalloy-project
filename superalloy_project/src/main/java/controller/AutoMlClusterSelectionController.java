package controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import python.AutoMlClusterSelection;
import service.AutoMlClusterSelectionService;
import service.DataService;

import javax.servlet.http.HttpServletRequest;

@Controller
@RequestMapping("/AutoMlClusterSelection")
public class AutoMlClusterSelectionController {
    @Autowired
    private DataService dataService;
    @Autowired
    private AutoMlClusterSelectionService autoMlClusterSelectionService;
    @Autowired
    private AutoMlClusterSelection autoMlClusterSelection;

    //计算元特征
    @RequestMapping("/MetaFeatures")
    public @ResponseBody
    String MetaFeatures(HttpServletRequest request, Model model,
                        @RequestParam("eCode") String eCode){
        String filename = dataService.getRecordByECode(eCode);
        System.out.println("filename:"+filename+"eCode"+eCode);
        String json = autoMlClusterSelectionService.MetaFeatures(filename);
        autoMlClusterSelectionService.Recommendation_UIM(filename);
        System.out.println(json);
        return json;
    }

//    @RequestMapping(value ="/filterOne",method = GET)
//    /**
//     *@Description:
//     *@Param: [request, model, eCode]
//     *@Return: java.lang.String
//     */
//    public @ResponseBody
//    String filter1(HttpServletRequest request, Model model,
//                   @RequestParam("eCode") String eCode) {
//        String fileName=dataService.getRecordByECode(eCode);
//        String json=filterService.filterServiceOne(fileName);
//        System.out.println(json);
//        return json;
//    }

//    //运行基于用户、物品、模型的推荐算法
//    @RequestMapping("/Recommend_UIM")
//    public @ResponseBody
//    String Recommend_UIM(HttpServletRequest request,Model model,
//                       @RequestParam("eCode") String eCode){
//        String filename = dataService.getRecordByECode(eCode);
//        String json = autoMlClusterSelectionService.MetaFeatures(filename);
////        autoMlClusterSelectionService.Recommendation_UIM(filename);
////        System.out.println(json);
//        return json;
//    }

    //取出基于用户的推荐算法
    @RequestMapping("/Recommend_User")
    public @ResponseBody
    String Recommend_User(HttpServletRequest request, Model model,
                              @RequestParam("eCode") String eCode){
        String filename = dataService.getRecordByECode(eCode);
        String json = autoMlClusterSelectionService.Recommendation_getUser(filename);
        System.out.println(json);
        return json;
    }

    //取出基于物品的推荐算法
    @RequestMapping("/Recommend_Item")
    public @ResponseBody
    String Recommend_Item(HttpServletRequest request, Model model,
                          @RequestParam("eCode") String eCode){
        String filename = dataService.getRecordByECode(eCode);
        String json = autoMlClusterSelectionService.Recommendation_getItem(filename);
        System.out.println(json);
        return json;
    }

    //取出基于模型的推荐算法
    @RequestMapping("/Recommend_Model")
    public @ResponseBody
    String Recommend_Model(HttpServletRequest request, Model model,
                          @RequestParam("eCode") String eCode){
        String filename = dataService.getRecordByECode(eCode);
        String json = autoMlClusterSelectionService.Recommendation_getModel(filename);
        System.out.println(json);
        return json;
    }

    //得到基于RF的推荐算法
    @RequestMapping("/Recommend_RF")
    public @ResponseBody
    String Recommend_RF(HttpServletRequest request, Model model,
                          @RequestParam("eCode") String eCode){
        String filename = dataService.getRecordByECode(eCode);
        String json = autoMlClusterSelectionService.Recommendation_RF(filename);

        System.out.println(json);
        return json;
    }

    //利用贝叶斯优化计算聚类算法
    @RequestMapping("/ClusterModel")
    public @ResponseBody
    String ClusterModel(HttpServletRequest request, Model model,
                            @RequestParam("eCode") String eCode, @RequestParam("alg") String alg){
        String filename = dataService.getRecordByECode(eCode);
        String json = autoMlClusterSelectionService.ClusterModel(filename,alg);

        System.out.println(json);
        return json;
    }
}
