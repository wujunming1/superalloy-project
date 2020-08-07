package service;

import com.alibaba.fastjson.JSONObject;
import com.fasterxml.jackson.databind.jsonFormatVisitors.JsonObjectFormatVisitor;
import dao.AutoMlClusterSelectionDao;
import dao.DataDao;
import entity.ClusterModelEntity;
import entity.MetaFeatureEntity;
import entity.RecommendResultEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import python.AutoMlClusterSelection;

@Service
public class AutoMlClusterSelectionService {
    @Autowired
    private DataDao dataDao;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private AutoMlClusterSelectionDao autoMlClusterSelectionDao;
    @Autowired
    private AutoMlClusterSelection autoMlClusterSelection;

    //第一次计算该数据集的元特征
    public String MetaFeatures(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        autoMlClusterSelectionDao.MetaFeature_Clean(username,filename);
        MetaFeature_Run(username,filename);
        MetaFeatureEntity result = autoMlClusterSelectionDao.MetaFeature_getResult(username,filename);
//        result.setHeader(autoMlClusterSelection.getHeader(filename));
        return JSONObject.toJSONString(result);
    }

    //运用python文件计算数据集的元特征
    private void MetaFeature_Run(String username,String filename){
        autoMlClusterSelection.MetaFeature_Run(username,filename);
    }

    //第一次运行基于用户、物品、模型的推荐算法
    public void Recommendation_UIM(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        autoMlClusterSelectionDao.Recommendation_Clean(username,filename);
        Recommendation_UIM_Run(username,filename);
    }

    //运用python文件对基于用户、物品、模型的运算
    private void Recommendation_UIM_Run(String username, String filename){
        autoMlClusterSelection.Recommendation_UIM_Run(username,filename);
    }

    //得到基于用户
    public String Recommendation_getUser(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        RecommendResultEntity User = autoMlClusterSelectionDao.Recommendation_getUserResult(username,filename);
        return JSONObject.toJSONString(User);
    }

    //得到基于物品
    public String Recommendation_getItem(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        RecommendResultEntity Item = autoMlClusterSelectionDao.Recommendation_getItemResult(username,filename);
        return JSONObject.toJSONString(Item);
    }

    //得到基于模型
    public String Recommendation_getModel(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        RecommendResultEntity Model = autoMlClusterSelectionDao.Recommendation_getModelResult(username,filename);
        return JSONObject.toJSONString(Model);
    }

    //第一次运行基于RF的推荐算法
    public String Recommendation_RF(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        autoMlClusterSelectionDao.Recommendation_RF_Clean(username,filename);
        Recommendation_RF_RUN(username,filename);
        RecommendResultEntity Model = autoMlClusterSelectionDao.Recommendation_getRFResult(username,filename);
        return JSONObject.toJSONString(Model);
    }

    //运用python文件对基于用户、物品、模型的运算
    private void Recommendation_RF_RUN(String username, String filename){
        autoMlClusterSelection.Recommendation_RF_RUN(username,filename);
    }

    //得到基于RF
    public String Recommendation_getRF(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        RecommendResultEntity Model = autoMlClusterSelectionDao.Recommendation_getRFResult(username,filename);
        return JSONObject.toJSONString(Model);
    }

    //第一次对该数据集进行贝叶斯优化的聚类算法运算
    public String ClusterModel(String filename, String alg){
        String username = sysAccountService.getLoginAccount().getUsername();
        autoMlClusterSelectionDao.ClusterModel_Clean(username,filename,alg);
        ClusterModel_Run(username,filename,alg);
        ClusterModelEntity result = autoMlClusterSelectionDao.ClusterModel_getResult(username,filename,alg);

        return JSONObject.toJSONString(result);
    }

    //运行贝叶斯优化的聚类算法
    private void ClusterModel_Run(String username, String filename, String alg){
        autoMlClusterSelection.ClusterModel_Run(username,filename,alg);
    }

}
