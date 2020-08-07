package service;

import com.alibaba.fastjson.JSONObject;
import dao.DataDao;
import dao.FilterDao;
import entity.RemainFeatureFour;
import entity.RemainFeatureOne;
import entity.RemainFeatureThree;
import entity.RemainFeatureTwo;
import model.FeatureTwoModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import python.PythonRun;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by piki on 2017/11/26.
 */
@Service
public class FilterService {
    @Autowired
    private DataDao dataDao;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private FilterDao filterDao;
    @Autowired
    private PythonRun pythonRun;
    @Autowired
    private ExcelAnalysisService excelAnalysisService;
    @Value("${python.cor_coeff}")
    private String cor_coeff;



    /**
     *@Description: 第一层特征选择
     *@Param: [fileName]
     *@Return: java.lang.String
     */
    public String filterServiceOne(String fileName){
        String username=sysAccountService.getLoginAccount().getUsername();
        filterDao.cleanFilter(username);
        filterRun(username,fileName,"feature1.py");
        RemainFeatureOne result=filterDao.getRemainOne(username);
        result.setHeader(pythonRun.getHeader(fileName));
        return JSONObject.toJSONString(result);
    }
    /**
     *@Description: 第二次特征选择
     *@Param: [fileName, remain]
     *@Return: java.lang.String
     */
    public String filterServiceTwo(String fileName, int[] remain){
        String username=sysAccountService.getLoginAccount().getUsername();
        filterRun(username,fileName,remain,"feature2.py");
        RemainFeatureTwo result=filterDao.getRemainTwo(username);
        result.setHeader(pythonRun.getHeader(fileName));

        ArrayList<ArrayList<Object>> relation= excelAnalysisService.readExcel(cor_coeff);
        result.setRelation(relation);
        FeatureTwoModel model=new FeatureTwoModel(result);
        return JSONObject.toJSONString(result);
    }
    /**
     *@Description: 第三层特征选择
     *@Param: [fileName, remain]
     *@Return: java.lang.String
     */
    public String filterServiceThree(String fileName, int[] remain){
        String username=sysAccountService.getLoginAccount().getUsername();
        filterRun(username,fileName,remain,"feature3.py");
        RemainFeatureThree result=filterDao.getRemainThree(username);
//        System.out.println((result.getShow3().toArray()));
        result.setHeader(pythonRun.getHeader(fileName));
        return JSONObject.toJSONString(result);
    }

    /**
     *@Description: 第四层特征选择
     *@Param: [fileName, remain]
     *@Return: java.lang.String
     */
    public String filterServiceFour(String fileName, int[] remain){
        String username=sysAccountService.getLoginAccount().getUsername();
        filterRun(username,fileName,remain,"feature4.py");
//        filterRun(username,fileName,"feature1.py");

        RemainFeatureFour result=filterDao.getRemainFour(username);
        result.setHeader(pythonRun.getHeader(fileName));
        return JSONObject.toJSONString(result);
    }

    /**
     *@Description: 一键特征选择
     *@Param: [fileName]
     *@Return: java.lang.String
     */
    public String oneKeyFilter(String fileName){
        String username=sysAccountService.getLoginAccount().getUsername();
        filterRun(username,fileName,"feature1.py");
        pythonRun.runpython(username,fileName,"feature2.py","null");
        pythonRun.runpython(username,fileName,"feature3.py","null");
        pythonRun.runpython(username,fileName,"feature4.py","null");
        RemainFeatureFour result=filterDao.getRemainFour(username);
        result.setHeader(pythonRun.getHeader(fileName));
        return JSONObject.toJSONString(result);
    }

    private void filterRun(String username,String fileName, int[] remain, String algorithm){
        StringBuilder strbRemain=new StringBuilder();
        if(remain[0]==-1){
            strbRemain=new StringBuilder("[null]");
        }else{
            strbRemain= new StringBuilder(Arrays.toString(remain));
        }
        pythonRun.runpython(username,fileName,algorithm,strbRemain.substring(1,strbRemain.length()-1));
    }

    private void filterRun(String username,String fileName, String algorithm){

        pythonRun.runpython(username,fileName,algorithm);
    }


    public boolean DBIsEmpty(){
        return filterDao.getDBIsEmpty();
    }

    public void cleanFilterDB(){
        filterDao.cleanFilter(sysAccountService.getLoginAccount().getUsername());
    }
}
