package service;

import com.alibaba.fastjson.JSONObject;
import entity.PreprocessingEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import python.CreepPrediction;
import python.Preprocessing;
@Service
public class CreepPredictionService {
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private CreepPrediction creepPrediction;
    public String ClusterAlloy(String clusterNum, String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        creepPrediction.ClusterSuperalloy(clusterNum, filename, username);
        return "success";
    }
    public String DCSA_analysis(String filename,String kernel, String C,
                                String gpr_alpha, String optimized_num, String estimators,
                                String maxDepth, String lr_alpha, String rr_alpha ){
        String username = sysAccountService.getLoginAccount().getUsername();
        creepPrediction.DCSA_analysis(filename, username, kernel, C, gpr_alpha,
                optimized_num, estimators, maxDepth, lr_alpha, rr_alpha);
        return "success";
    }
}
