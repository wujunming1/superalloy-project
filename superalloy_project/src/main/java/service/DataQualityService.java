package service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import python.DataQualityPy;

import java.io.File;
import java.util.ArrayList;

/**
 * Created by piki on 2017/9/21.
 */
@Service
public class DataQualityService {
    @Value("${file.quality}")
    private String qualityPath;

    @Autowired
    DataQualityPy dataQualityPy;
    @Autowired
    ExcelAnalysisService excelAnalysisService;

    /**质量检测服务
     *@Description:
     *@Param: [serverFile, username]
     *@Return: java.util.ArrayList<java.util.ArrayList<java.lang.Object>>
     */
    public ArrayList<ArrayList<Object>> getDataQuality(String serverFile,String username){

        File file = new File(qualityPath+username+".xlsx");
        if (file.isFile() && file.exists()) {
            file.delete();
        }
        dataQualityPy.dataQuality(serverFile,username);//生成质量检测文件
        if(file.exists()){//根据质量检测文件是否存在来判断数据集是否可以进行质量检测
            ArrayList<ArrayList<Object>> dataQuality=excelAnalysisService.readExcel(qualityPath+username+".xlsx");
            dataQuality.get(1).set(0,"\\u6837\\u672c\\u6570");
            dataQuality.get(2).set(0,"\\u5e73\\u5747\\u6570");
            dataQuality.get(3).set(0,"\\u6807\\u51c6\\u5dee");
            dataQuality.get(4).set(0,"\\u6700\\u5c0f\\u503c");
            dataQuality.get(5).set(0,"\\u56db\\u5206\\u4f4d\\u6570Q1");
            dataQuality.get(6).set(0,"\\u56db\\u5206\\u4f4d\\u6570Q2");
            dataQuality.get(7).set(0,"\\u56db\\u5206\\u4f4d\\u6570Q3");
            dataQuality.get(8).set(0,"\\u6700\\u5927\\u503c");
            dataQuality.get(9).set(0,"\\u504f\\u5ea6");
            dataQuality.get(10).set(0,"\\u6781\\u5dee");
            dataQuality.get(11).set(0,"\\u51e0\\u4f55\\u5e73\\u5747\\u6570");
            dataQuality.get(12).set(0,"\\u56db\\u5206\\u4f4d\\u95f4\\u8ddd");
            dataQuality.get(13).set(0,"\\u4e0a\\u9650");
            dataQuality.get(14).set(0,"\\u4e0b\\u9650");
            return dataQuality;
        }
        else{//如果不能进行质量检测，则返回null
            return null;
        }
    }
}
