package controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import python.AutoMlClassifierSelection;
import service.AutoMlClassifierSelectionService;
import service.DataService;

import javax.servlet.http.HttpServletRequest;

@Controller
@RequestMapping("/AutoMlClassifierSelection")
public class AutoMlClassifierSelectionController {
    @Autowired
    private DataService dataService;
    @Autowired
    private AutoMlClassifierSelectionService autoMlClassifierSelectionService;
    @Autowired
    private AutoMlClassifierSelection autoMlClassifierSelection;

    //计算元特征
    @RequestMapping("/MetaFeatures")
    public @ResponseBody
    String MetaFeatures(HttpServletRequest request, Model model,
                        @RequestParam("eCode") String eCode){
        String filename = dataService.getRecordByECode(eCode);
        System.out.println("filename:"+filename+"eCode"+eCode);
        String json = autoMlClassifierSelectionService.MetaFeatures(filename);
        autoMlClassifierSelectionService.Recommendation_UIM(filename);
        System.out.println(json);
        return json;
    }



    //取出基于用户的推荐算法
    @RequestMapping("/Recommend_User")
    public @ResponseBody
    String Recommend_User(HttpServletRequest request, Model model,
                              @RequestParam("eCode") String eCode){
        String filename = dataService.getRecordByECode(eCode);
        String json = autoMlClassifierSelectionService.Recommendation_getUser(filename);
        System.out.println(json);
        return json;
    }

    //取出基于物品的推荐算法
    @RequestMapping("/Recommend_Item")
    public @ResponseBody
    String Recommend_Item(HttpServletRequest request, Model model,
                          @RequestParam("eCode") String eCode){
        String filename = dataService.getRecordByECode(eCode);
        String json = autoMlClassifierSelectionService.Recommendation_getItem(filename);
        System.out.println(json);
        return json;
    }

    //取出基于模型的推荐算法
    @RequestMapping("/Recommend_Model")
    public @ResponseBody
    String Recommend_Model(HttpServletRequest request, Model model,
                          @RequestParam("eCode") String eCode){
        String filename = dataService.getRecordByECode(eCode);
        String json = autoMlClassifierSelectionService.Recommendation_getModel(filename);
        System.out.println(json);
        return json;
    }

    //得到基于RF的推荐算法
    @RequestMapping("/Recommend_RF")
    public @ResponseBody
    String Recommend_RF(HttpServletRequest request, Model model,
                          @RequestParam("eCode") String eCode){
        String filename = dataService.getRecordByECode(eCode);
        String json = autoMlClassifierSelectionService.Recommendation_RF(filename);

        System.out.println(json);
        return json;
    }

    //利用贝叶斯优化计算分类算法
    @RequestMapping("/ClassifierModel")
    public @ResponseBody
    String ClassifierModel(HttpServletRequest request, Model model,
                            @RequestParam("eCode") String eCode, @RequestParam("alg") String alg){
        String filename = dataService.getRecordByECode(eCode);
        String json = autoMlClassifierSelectionService.ClassifierModel(filename,alg);

        System.out.println(json);
        return json;
    }
}
