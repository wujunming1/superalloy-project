package service;
import com.alibaba.fastjson.JSONObject;
import dao.DataDao;
import dao.ClusterIntegrationDao;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import entity.ClusterIntegrationEntity;
import entity.ClusterStatusEntity;
import java.util.List;
import python.ClusterIntegration;
@Service
public class ClusterIntegrationService {
    @Autowired
    private DataDao dataDao;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private ClusterIntegrationDao clusterIntegrationDao;
    @Autowired
    private  ClusterIntegration clusterIntegration;


    //    非自动式只需要得到聚类算法的结果
    public String getClusterResult(String fileName){
        String username=sysAccountService.getLoginAccount().getUsername();
        ClusterIntegrationEntity result=clusterIntegrationDao.getResult(username,fileName);

        return JSONObject.toJSONString(result);
    }
    //    任务完成表
    public String getClusterStatusComplete(){
        String username=sysAccountService.getLoginAccount().getUsername();
        List<ClusterStatusEntity> result=clusterIntegrationDao.getStatusComplete(username);

        return JSONObject.toJSONString(result);
    }
    //    任务完成表
    public String getClusterStatusRun(){
        String username=sysAccountService.getLoginAccount().getUsername();
        List<ClusterStatusEntity> result=clusterIntegrationDao.getStatusRun(username);

        return JSONObject.toJSONString(result);
    }
    //    任务完成表
    public String getClusterStatusWait(){
        String username=sysAccountService.getLoginAccount().getUsername();
        List<ClusterStatusEntity> result=clusterIntegrationDao.getStatusWait(username);

        return JSONObject.toJSONString(result);
    }
    //    运行自动式聚类算法
    private void AutoclusterRun(String username,String fileName){
        clusterIntegration.AutorunCluster(username,fileName);
    }
    private void AutoclusterDelete(String username,String fileName){
        clusterIntegration.AutoclusterDelete(username,fileName);
    }
    private void AutoclusterInsert(String username,String fileName){
        clusterIntegration.AutoclusterInsert(username,fileName);
    }
    public String AutoCluster(String fileName){
        String username=sysAccountService.getLoginAccount().getUsername();
        clusterIntegrationDao.Auto_cleanCluster(username,fileName);
        AutoclusterRun(username,fileName);
        ClusterIntegrationEntity result=clusterIntegrationDao.getResult(username,fileName);
        return JSONObject.toJSONString(result);
    }
    public String Clusterinsert(String fileName){
        String username=sysAccountService.getLoginAccount().getUsername();
        clusterIntegrationDao.Auto_cleanStatus(username,fileName);
        AutoclusterInsert(username,fileName);
        String result="完成";
        System.out.println(result);
        return JSONObject.toJSONString(result);
    }
    public String Clusterdelete(String fileName){
        String username=sysAccountService.getLoginAccount().getUsername();
        clusterIntegrationDao.Auto_cleanStatus(username,fileName);
        AutoclusterDelete(username,fileName);
        String result="完成";
        System.out.println(result);
        return JSONObject.toJSONString(result);
    }
    public String StartDelete(String fileName){
        String username=sysAccountService.getLoginAccount().getUsername();
        clusterIntegrationDao.Auto_cleanStatus(username,fileName);
        String result="完成";
        System.out.println(result);
        return JSONObject.toJSONString(result);
    }

}
