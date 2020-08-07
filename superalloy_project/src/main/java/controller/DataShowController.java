package controller;

import com.alibaba.fastjson.JSONObject;
import entity.SysAccount;
import model.insert.DataDescriptionModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.bson.Document;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import service.DataQualityService;
import service.DataService;
import service.SysAccountService;

import javax.servlet.http.HttpServletRequest;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.springframework.web.bind.annotation.RequestMethod.GET;

/**
 * Created by piki on 2018/4/22.
 */
@Controller
@RequestMapping("/dataShow")
public class DataShowController {
    @Value("${file.data}")
    private String dataPath;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private DataService dataService;
    @Autowired
    private DataQualityService dataQualityService;

    @RequestMapping(value ="/getDataShow",method = GET)
    public @ResponseBody
    String getDataShow(HttpServletRequest request, Model model,
                   @RequestParam("eCode") String eCode) {
        JSONObject jsonObject = new JSONObject();
        SysAccount account = sysAccountService.getLoginAccount();
        List<Document> data=dataService.getDataByECode(eCode);
        String uploadRootPath = dataPath+account.getUsername();
        File uploadRootDir = new File(uploadRootPath);
        String record=dataService.getRecordByECode(eCode);
        File serverFile = new File(uploadRootDir.getAbsolutePath()
                + File.separator + record);
        ArrayList<ArrayList<Object>> dataQuality=dataQualityService.getDataQuality(serverFile.toString(),account.getUsername());

        jsonObject.put("dataResult",JSONObject.toJSONString(data));
        jsonObject.put("dataQuality",JSONObject.toJSONString(dataQuality));

        return jsonObject.toJSONString();
    }

    @RequestMapping(value ="/getDataShowDescribe",method = GET,produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getDataShowDescribe(HttpServletRequest request, Model model,
                   @RequestParam("eCode") String eCode) {
        JSONObject jsonObject = new JSONObject();
        DataDescriptionModel description=dataService.getDataDescriptionByECode(eCode);
        jsonObject.put("dataDescription",JSONObject.toJSONString(description));

        return jsonObject.toJSONString();
    }
}
