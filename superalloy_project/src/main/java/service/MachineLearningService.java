package service;

import com.alibaba.fastjson.JSONObject;
import dao.DataDao;
import dao.FilterDao;
import entity.*;
import mapper.AlgorithmEntityMapper;
import model.FeatureTwoModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import python.AlgorithmService;
import python.PythonRun;

import java.io.File;
import java.net.URL;
import java.util.Arrays;

/**
 * Created by piki on 2017/11/26.
 */
@Service
public class MachineLearningService {

    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private FilterDao filterDao;
    @Autowired
    private PythonRun pythonRun;
    @Autowired
    private DataService dataService;
    @Autowired
    private AlgorithmService algorithmService;
    @Autowired
    private AlgorithmEntityMapper algorithmEntityMapper;

    @Value("${file.data}")
    private String filePath;
    @Value("${python.algorithm}")
    private String algorithmPath;
    @Value("${python.picPath}")
    private String picImagePath;

    public String machineLearning(String eCode,String algoId){
        /**
         *@Description: 
         *@Param: [eCode, algoId]
         *@Return: java.lang.String
         */
        String username=sysAccountService.getLoginAccount().getUsername();
        String fileName=dataService.getRecordByECode(eCode);
        AlgorithmEntity algo=algorithmEntityMapper.getAlgoById(algoId);
        URL url=null;
        try{
            url=new URL(this.getClass().getClassLoader().getResource("/"),"../..");
        }catch(Exception e){
            e.printStackTrace();
            return "";
        }
        String picPath=url.getPath().substring(1)+picImagePath+username+"\\";
        File picPathDir = new File(picPath);
        if (!picPathDir.exists()) {
            picPathDir.mkdirs();
        }
        picPath=algorithmService.runAlgorithm(filePath+username+"\\"+fileName,algo,picPath);

        return picImagePath+username+"\\"+algo.getAlgoName()+".jpg";
    }
}
