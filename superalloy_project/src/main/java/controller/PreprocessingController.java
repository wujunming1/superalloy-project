package controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import service.DataService;
import service.PreprocessingService;

import javax.servlet.http.HttpServletRequest;

@Controller
@RequestMapping("/Preprocessing")
public class PreprocessingController {

    @Autowired
    private DataService dataService;
    @Autowired
    private PreprocessingService PreprocessingService;

    @RequestMapping("/Encoding")
    public @ResponseBody
    /**
     *@Description: 数据预处理数据编码接口
     *@Param: Cluster_ALG：数据预处理的方法
     *@Param: onehot：需要进行独热编码的列
     *@Param: numerical：需要进行数值化编码的列
     *@Param: numerical_dict：数值化编码映射关系
     *@Return: java.lang.String
     */
    String Encoding(HttpServletRequest request, Model model,
                    @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
                    @RequestParam("onehot") String onehot, @RequestParam("numerical") String numerical,
                    @RequestParam("numerical_dict") String numerical_dict){
        if(onehot.equals(""))
            onehot = "none";
        if(numerical.equals(""))
            numerical = "none";
        if(numerical_dict.equals(""))
            numerical_dict = "none";
        String filename = dataService.getRecordByECode(eCode);
        String json = PreprocessingService.Encode(filename,Cluster_ALG,onehot,numerical,numerical_dict);
        System.out.println(json);
        return json;
    }

    @RequestMapping("/Null_check")
    public @ResponseBody
    /**
     *@Description: 数据预处理数据编码接口
     *@Param: Cluster_ALG：数据预处理的方法
     *@Param: null_type：进行空值检测的空值类型
     *@Return: java.lang.String
     */
    String Null_check(HttpServletRequest request, Model model,
                    @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
                    @RequestParam("null_type") String null_type){
        String filename = dataService.getRecordByECode(eCode);
        String json = PreprocessingService.Nullcheck(filename,Cluster_ALG,null_type);
        System.out.println(json);
        return json;
    }

    @RequestMapping("/Null_value")
    public @ResponseBody
    /**
     *@Description: 数据预处理数据编码接口
     *@Param: Cluster_ALG：数据预处理的方法
     *@Param: null_type：空值类型
     *@Return: java.lang.String
     */
    String Null_valuek(HttpServletRequest request, Model model,
                      @RequestParam("eCode") String eCode, @RequestParam("Cluster_ALG") String Cluster_ALG,
                      @RequestParam("null_type") String null_type, @RequestParam("null_method") String null_method,
                       @RequestParam("null_para") String null_para){
        String filename = dataService.getRecordByECode(eCode);
        String json = PreprocessingService.Nullvalue(filename,Cluster_ALG,null_type,null_method,null_para);
        System.out.println(json);
        return json;
    }
}
