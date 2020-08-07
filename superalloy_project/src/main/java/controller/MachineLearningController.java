package controller;

import dao.DataDao;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import python.PythonRun;
import service.*;

import javax.servlet.http.HttpServletRequest;
import java.util.Arrays;
import java.util.List;

import static org.springframework.web.bind.annotation.RequestMethod.GET;

/**
 * Created by piki on 2017/10/24.
 */

@Controller
@RequestMapping("/machineLearning")
public class MachineLearningController {
    @Autowired
    private MachineLearningService machineLearningService;
    @Autowired
    private DataService dataService;

    @Autowired
    private FilterService filterService;


    /**
     *@Description: 机器学习接口
     *@Param: [request, model, eCode, algoId]
     *@Return: java.lang.String
     */
    @RequestMapping(value ="/algo",method = GET)
    public @ResponseBody
    String algo(HttpServletRequest request, Model model,
                   @RequestParam("eCode") String eCode,@RequestParam("algoId") String algoId) {
        String picPath=machineLearningService.machineLearning(eCode,algoId);
        System.out.println(picPath);
        return picPath;
    }

}
