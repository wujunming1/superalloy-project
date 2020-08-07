package controller;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import dao.DataDao;
import entity.SysAccount;
import mapper.PatentDataModelMapper;
import model.insert.PatentDataModel;
import org.bson.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;
import service.DataService;
import service.ExcelAnalysisService;
import service.SysAccountService;

import javax.servlet.http.HttpServletRequest;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.*;

import static org.springframework.web.bind.annotation.RequestMethod.GET;
import static org.springframework.web.bind.annotation.RequestMethod.POST;

/**
 * Created by piki on 2017/9/17.
 */

@Controller
@RequestMapping("/data")
public class Data {
    @Autowired
    private ExcelAnalysisService excelAnalysisService;
    @Autowired
    private DataDao dataDao;
    @Autowired
    private DataService dataService;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private PatentDataModelMapper patentDataModelMapper;
    @RequestMapping(value = "/deletePatent",method = RequestMethod.POST)
    @ResponseBody
    public String delete_p(HttpServletRequest request, Model model){
        //批量删除选中行的专利数据
        String[] index_list = request.getParameterValues("ids");
        System.out.println(index_list);
        for(int i=0;i<index_list.length;i++){
            Integer alloy_index = Integer.valueOf(index_list[i]);
            patentDataModelMapper.deletePatent(alloy_index);
        }
        JSONObject result = new JSONObject();
        result.put("state", "success");
        return result.toJSONString();
    }
    @RequestMapping(value ="/record",method = GET)
    public @ResponseBody List<String> getUserDataRecord( String username, HttpServletRequest request, Model model) {
        List<String > recordList=dataService.getUserDataRecord("piki");

        return recordList;
    }

    @RequestMapping(value ="/getData",method = GET)
    public @ResponseBody
    String  getData(@RequestParam("eCode") String eCode, HttpServletRequest request, Model model) {

        List<Document> recordList=dataService.getDataByECode(eCode);
        JSONObject json=new JSONObject();
        // JSONObject data=new JSONObject();
        Set<String> title=new HashSet<String>();
        List<String> data=new ArrayList<String>();
        for(Document document:recordList){
            Set<String> keys=document.keySet();
            title.addAll(keys);
            JSONObject row = new JSONObject();
            for(String key:keys){
                row.put(key,document.get(key));
            }
            data.add(row.toJSONString());
        }
        title.remove("record");
        title.remove("username");
        title.remove("_id");
        Map<String,Object> m = new HashMap<String, Object>();
        m.put("title",title.toArray());
        json.put("title", JSON.toJSON(title));
        json.put("data",JSON.toJSON(data));
        System.out.println(json.toJSONString());
        return json.toJSONString();
    }

    @RequestMapping(value ="/dataInfo",method = POST)
    public @ResponseBody String uploadDataInfo( String username, HttpServletRequest request, Model model) {

        return null;
    }
    //通过调用service中的search方法
    @RequestMapping(value ="/search",method = GET)
    public @ResponseBody List search( String condition, HttpServletRequest request, Model model) {
        List recordList=dataService.search(condition);
        return recordList;
    }

}
