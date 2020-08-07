package controller;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import service.ClusterIntegrationService;
import service.DataService;

import javax.servlet.http.HttpServletRequest;
import java.util.List;

@Controller
@RequestMapping("/clusterintegration")
public class ClusterIntegrationController {
    @Autowired
    private DataService dataService;
    @Autowired
    private ClusterIntegrationService clusterIntegrationService;
    //    结果展示
    @RequestMapping(value = "/NonAutoAlgorithm")
    public @ResponseBody
    String ClusterAlogorithm(HttpServletRequest request, Model model,
                             @RequestParam("data_name") String data_name,@RequestParam("username") String username){
        System.out.println(data_name);
        String json=clusterIntegrationService.getClusterResult(data_name);


        return json;
    }
    //    任务完成表
    @RequestMapping(value = "/StatusComplete")
    public @ResponseBody
    String ClusterStatusComplete(HttpServletRequest request, Model model,
                             @RequestParam("eCode") String eCode){
        String fileName = dataService.getRecordByECode(eCode);
        String json=clusterIntegrationService.getClusterStatusComplete();


        return json;
    }
    //    任务运行表
    @RequestMapping(value = "/StatusRun")
    public @ResponseBody
    String ClusterStatusRun(HttpServletRequest request, Model model,
                         @RequestParam("eCode") String eCode){
        String fileName = dataService.getRecordByECode(eCode);
        String json=clusterIntegrationService.getClusterStatusRun();

        return json;
    }
    //    新上传任务表
    @RequestMapping(value = "/StatusWait")
    public @ResponseBody
    String ClusterStatusWait(HttpServletRequest request, Model model,
                            @RequestParam("eCode") String eCode){
        String fileName = dataService.getRecordByECode(eCode);
        String json=clusterIntegrationService.getClusterStatusWait();

        return json;
    }
    //    非自动式
    @RequestMapping(value = "/clusterintegration")
    public @ResponseBody
    String ClusterIntegration(HttpServletRequest request, Model model,
                              @RequestParam("data_name") String data_name){
        String json=clusterIntegrationService.AutoCluster(data_name);


        return json;
    }
    //    上传
    @RequestMapping(value = "/clusterinsert")
    public @ResponseBody
    String ClusterInsert(HttpServletRequest request, Model model,
                         @RequestParam("eCode") String eCode){
        String fileName = dataService.getRecordByECode(eCode);
        String json=clusterIntegrationService.Clusterinsert(fileName);


        return json;
    }
    //    删除
    @RequestMapping(value = "/clusterdelete")
    public @ResponseBody
    String ClusterDelete(HttpServletRequest request, Model model,
                         @RequestParam("username") String username,@RequestParam("data_name") String data_name) {
        String json = clusterIntegrationService.Clusterdelete(data_name);


        return json;
    }

    @RequestMapping(value = "/startdelete")
    public @ResponseBody
    String StartDelete(HttpServletRequest request, Model model,
                        @RequestParam("data_name") String data_name){
        String json=clusterIntegrationService.StartDelete(data_name);


        return json;
    }
}
