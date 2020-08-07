package service;

import com.alibaba.fastjson.JSONObject;
import entity.PreprocessingEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import python.Preprocessing;

@Service
public class PreprocessingService {
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private Preprocessing preprocessing;
    @Autowired
    private dao.PreprocessingDao PreprocessingDao;

    public String Encode(String filename,String Cluster_ALG,String onehot,
                         String numerical,String numerical_dict){
        String username = sysAccountService.getLoginAccount().getUsername();
        PreprocessingDao.AlgModel_Clean(username,filename,Cluster_ALG);
        preprocessing.Encode_Run(username, filename, Cluster_ALG, onehot, numerical, numerical_dict);
        PreprocessingEntity result = PreprocessingDao.AlgModel_getResult(username,filename,Cluster_ALG);
        return JSONObject.toJSONString(result);
    }
    public String Nullcheck(String filename,String Cluster_ALG,String null_type){
        String username = sysAccountService.getLoginAccount().getUsername();
        PreprocessingDao.AlgModel_Clean(username,filename,Cluster_ALG);
        preprocessing.Nullcheck_Run(username, filename, Cluster_ALG, null_type);
        PreprocessingEntity result = PreprocessingDao.AlgModel_getResult(username,filename,Cluster_ALG);
        return JSONObject.toJSONString(result);
    }
    public String Nullvalue(String filename,String Cluster_ALG,String null_type,String null_method,String null_para){
        String username = sysAccountService.getLoginAccount().getUsername();
        PreprocessingDao.AlgModel_Clean(username,filename,Cluster_ALG);
        preprocessing.Nullvalue_Run(username, filename, Cluster_ALG, null_type,null_method,null_para);
        PreprocessingEntity result = PreprocessingDao.AlgModel_getResult(username,filename,Cluster_ALG);
        return JSONObject.toJSONString(result);
    }
}
