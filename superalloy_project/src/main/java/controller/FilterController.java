package controller;

import dao.DataDao;
import org.python.antlr.ast.Str;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import service.DataService;
import service.ExcelAnalysisService;
import python.PythonRun;
import service.FilterService;
import service.SysAccountService;

import javax.servlet.http.HttpServletRequest;

import java.util.Arrays;
import java.util.List;

import static org.springframework.web.bind.annotation.RequestMethod.GET;
/**
 * Created by piki on 2017/10/24.
 */

@Controller
@RequestMapping("/filter")
public class FilterController {
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private ExcelAnalysisService excelAnalysisService;
    @Autowired
    private DataDao dataDao;
    @Autowired
    private DataService dataService;
    @Autowired
    private PythonRun pythonRun;
    @Autowired
    private FilterService filterService;

    
    @RequestMapping(value ="/getHeader",method = GET)
    /**
     *@Description: 获取文档表头
     *@Param: [request, model, eCode]
     *@Return: java.util.List<java.lang.String>
     */
    public @ResponseBody
    List<String> getHeader(HttpServletRequest request, Model model,
                   @RequestParam("eCode") String eCode) {

        return pythonRun.getHeaderByECode(eCode);
    }

    @RequestMapping(value ="/filterOne",method = GET)
    /**
     *@Description: 
     *@Param: [request, model, eCode]
     *@Return: java.lang.String
     */
    public @ResponseBody
    String filter1(HttpServletRequest request, Model model,
                   @RequestParam("eCode") String eCode) {
        String fileName=dataService.getRecordByECode(eCode);
        String json=filterService.filterServiceOne(fileName);
        System.out.println(json);
        return json;
    }

    
    @RequestMapping(value ="/filterTwo",method = GET)
    /**
     *@Description:
     *@Param: [request, model, eCode, remain]
     *@Return: java.lang.String
     */
    public @ResponseBody
    String filterTwo(HttpServletRequest request, Model model,
                   @RequestParam("eCode") String eCode,@RequestParam("remain")int[] remain) {
//        if(remain[0]==-1){
//            remain=new int[0];
//        }
        String fileName=dataService.getRecordByECode(eCode);
        System.out.println(fileName);
        System.out.println(Arrays.toString(remain));

        String json= filterService.filterServiceTwo(fileName,remain);
        System.out.println(json);
        return json;
    }
    @RequestMapping(value ="/filterThree",method = GET)
    public @ResponseBody
    String filterThree(HttpServletRequest request, Model model,
                   @RequestParam("eCode") String eCode,@RequestParam("remain")int[] remain) {
//        if (remain[0] == -1) {
//            remain = new int[1];
//            remain[0] = 0;
//        }
        String fileName=dataService.getRecordByECode(eCode);

        String json = filterService.filterServiceThree(fileName, remain);
        System.out.println(json);
        return json;

    }

    @RequestMapping(value ="/filterFour",method = GET)
    public @ResponseBody
    String filterFour(HttpServletRequest request, Model model,
                     @RequestParam("eCode") String eCode,@RequestParam("remain")int[] remain) {
//        if(remain[0]==-1){
//            remain=new int[1];
//            remain[0]=0;
//        }
        String fileName=dataService.getRecordByECode(eCode);

        String json= filterService.filterServiceFour(fileName,remain);
        System.out.println(json);
        return json;
    }

    @RequestMapping(value ="/oneKeyFilter",method = GET)
    public @ResponseBody
    String oneKeyFilter(HttpServletRequest request, Model model,
                      @RequestParam("eCode") String eCode) {
        String fileName=dataService.getRecordByECode(eCode);
        String json= filterService.oneKeyFilter(fileName);
        System.out.println(json);
        return json;
    }
}
