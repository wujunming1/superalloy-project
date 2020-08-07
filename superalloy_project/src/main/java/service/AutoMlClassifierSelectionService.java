package service;

import com.alibaba.fastjson.JSONObject;
import dao.AutoMlClassifierSelectionDao;
import dao.DataDao;
import entity.ClassifierModelEntity;
import entity.MetaFeatureEntity;
import entity.RecommendResultEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import python.AutoMlClassifierSelection;

@Service
public class AutoMlClassifierSelectionService {
    @Autowired
    private DataDao dataDao;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private AutoMlClassifierSelectionDao AutoMlClassifierSelectionDao;
    @Autowired
    private AutoMlClassifierSelection AutoMlClassifierSelection;

    //第一次计算该数据集的元特征
    public String MetaFeatures(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        AutoMlClassifierSelectionDao.MetaFeature_Clean(username,filename);
        MetaFeature_Run(username,filename);
        MetaFeatureEntity result = AutoMlClassifierSelectionDao.MetaFeature_getResult(username,filename);
//        result.setHeader(AutoMlClassifierSelection.getHeader(filename));
        return JSONObject.toJSONString(result);
    }

    //运用python文件计算数据集的元特征
    private void MetaFeature_Run(String username,String filename){
        AutoMlClassifierSelection.MetaFeature_Run(username,filename);
    }

    //第一次运行基于用户、物品、模型的推荐算法
    public void Recommendation_UIM(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        AutoMlClassifierSelectionDao.Recommendation_Clean(username,filename);
        Recommendation_UIM_Run(username,filename);
    }

    //运用python文件对基于用户、物品、模型的运算
    private void Recommendation_UIM_Run(String username, String filename){
        AutoMlClassifierSelection.Recommendation_UIM_Run(username,filename);
    }

    //得到基于用户
    public String Recommendation_getUser(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        RecommendResultEntity User = AutoMlClassifierSelectionDao.Recommendation_getUserResult(username,filename);
        return JSONObject.toJSONString(User);
    }

    //得到基于物品
    public String Recommendation_getItem(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        RecommendResultEntity Item = AutoMlClassifierSelectionDao.Recommendation_getItemResult(username,filename);
        return JSONObject.toJSONString(Item);
    }

    //得到基于模型
    public String Recommendation_getModel(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        RecommendResultEntity Model = AutoMlClassifierSelectionDao.Recommendation_getModelResult(username,filename);
        return JSONObject.toJSONString(Model);
    }

    //第一次运行基于RF的推荐算法
    public String Recommendation_RF(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        AutoMlClassifierSelectionDao.Recommendation_RF_Clean(username,filename);
        Recommendation_RF_RUN(username,filename);
        RecommendResultEntity Model = AutoMlClassifierSelectionDao.Recommendation_getRFResult(username,filename);
        return JSONObject.toJSONString(Model);
    }

    //运用python文件对基于用户、物品、模型的运算
    private void Recommendation_RF_RUN(String username, String filename){
        AutoMlClassifierSelection.Recommendation_RF_RUN(username,filename);
    }

    //得到基于RF
    public String Recommendation_getRF(String filename){
        String username = sysAccountService.getLoginAccount().getUsername();
        RecommendResultEntity Model = AutoMlClassifierSelectionDao.Recommendation_getRFResult(username,filename);
        return JSONObject.toJSONString(Model);
    }

    //第一次对该数据集进行贝叶斯优化的分类算法运算
    public String ClassifierModel(String filename, String alg){
        String username = sysAccountService.getLoginAccount().getUsername();
        AutoMlClassifierSelectionDao.ClassifierModel_Clean(username,filename,alg);
        ClassifierModel_Run(username,filename,alg);
        ClassifierModelEntity result = AutoMlClassifierSelectionDao.ClassifierModel_getResult(username,filename,alg);

        return JSONObject.toJSONString(result);
    }

    //运行贝叶斯优化的分类算法
    private void ClassifierModel_Run(String username, String filename, String alg){
        AutoMlClassifierSelection.ClassifierModel_Run(username,filename,alg);
    }

}
