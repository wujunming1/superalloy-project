package controller;

import com.alibaba.fastjson.JSONObject;
import dao.DataRecordDao;
import entity.User;
import jnr.ffi.annotations.In;
import mapper.DataDescriptionModelMapper;
import mapper.LiteInfoModelMapper;
import mapper.PatentDataModelMapper;
import mapper.User_Upload_DataModelMapper;
import model.insert.DataDescriptionModel;
import model.insert.LiteInfoModel;
import model.insert.PatentDataModel;
import model.insert.User_Upload_DataModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import service.DataService;
import service.SysAccountService;

import javax.servlet.http.HttpServletRequest;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.springframework.web.bind.annotation.RequestMethod.GET;
import static org.springframework.web.bind.annotation.RequestMethod.POST;

@Controller
@RequestMapping("/search")
public class Search {
    @Autowired
    private DataRecordDao dataRecordDao;
    @Autowired
    private SysAccountService sysAccountService;
    //    @Autowired
//    private DataDescriptionModelMapper dataDescriptionModelMapper;
    @Autowired
    private DataDescriptionModelMapper dataDescriptionModelMapper;
    @Autowired
    private DataService dataService;
    @Autowired
    private LiteInfoModelMapper liteInfoModelMapper;
    @Autowired
    private PatentDataModelMapper patentDataModelMapper;
    @Autowired
    private User_Upload_DataModelMapper user_upload_dataModelMapper;
    @RequestMapping(value = "/getUploadDataInfo", method = GET, produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getUploadDataHeader(HttpServletRequest request, Model model) {
        //上传数据的描述信息
        //获取用户名
        String username = sysAccountService.getLoginAccount().getUsername();
        System.out.println("hello world!");
        List<User_Upload_DataModel> lists;
        if(username==null||username.length()==0){
            lists=null;
        }
        else{
            lists=user_upload_dataModelMapper.getUploadDataInfo(username);
        }
        return JSONObject.toJSONString(lists);
    }
    @RequestMapping(value = "/getRecord", method = GET, produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getHeader(HttpServletRequest request, Model model,
                     @RequestParam("keyword") String keyword) {
//        数据的描述信息
//        String userECode = sysAccountService.getLoginAccount().getUserECode();
//        System.out.println("用户ecode" + userECode);
        System.out.println("hello world!");
        List<DataDescriptionModel> lists;
            if (keyword.length() > 0) {
                keyword = "%" + keyword + "%";
                lists = dataDescriptionModelMapper.getDataDescriptionByRecordName(keyword);
            } else {
                lists = dataDescriptionModelMapper.getDataDescriptionList();
            }
        return JSONObject.toJSONString(lists);
    }
    @RequestMapping(value = "/getDataDesInfoByID", method =POST, produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getLiteInfoListByIDHeader(HttpServletRequest request, Model model) {
        String data_id =request.getParameter("relation_id");
        Integer description_id = Integer.valueOf(data_id);
        DataDescriptionModel descriptionModel =
                dataDescriptionModelMapper.getDataDescriptionByID(description_id);
        return JSONObject.toJSONString(descriptionModel);
    }
    @RequestMapping(value = "/getPatentRecord", method = GET, produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getPatentHeader(HttpServletRequest request, Model model) {
        System.out.println("hello world!");
        List<PatentDataModel> lists;
        lists = dataService.getPatentDataRecord();
        return JSONObject.toJSONString(lists);
    }
    @RequestMapping(value = "/getPatentDataByCondition", method = POST, produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getPatentByConditionHeader(HttpServletRequest request, Model model,
                           @RequestBody JSONObject jsondata) {
        System.out.println("hello world!");
        List<PatentDataModel> lists;
        String select_desciptor=jsondata.getString("select_descriptor");
        if(select_desciptor.equals("DL")){
            lists=patentDataModelMapper.getPatentDataList();
        }else {
            String lowlimit = jsondata.getString("lowlimit");
            String upperlimit = jsondata.getString("upperlimit");
            System.out.println(select_desciptor + lowlimit + upperlimit);
            Double lowlimit1 = Double.parseDouble(lowlimit);
            Double upperlimit1 = Double.parseDouble(upperlimit);//将从前端获取到的参数转化为Int类型
            lists = patentDataModelMapper.getPatentDataByCondition(select_desciptor,
                    lowlimit1, upperlimit1);
        }
        //将返回数据格式转换为json字符串
        return JSONObject.toJSONString(lists);
    }
    @RequestMapping(value = "/getWangLiteRecord", method = GET, produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getWangLiteHeader(HttpServletRequest request, Model model,
                           @RequestParam("record") String record) {
//        String userECode = sysAccountService.getLoginAccount().getUserECode();
//        System.out.println("用户ecode" + userECode);
        System.out.println("hello world!");
        List<DataDescriptionModel> lists;

        if (record.length() > 0) {
            record = "%" + record + "%";
            lists = dataDescriptionModelMapper.getWangLiteDataDescriptionByRecordName(record);
        } else {
            lists = dataDescriptionModelMapper.getWangLiteDataDescriptionList();
        }
        return JSONObject.toJSONString(lists);
    }
    @RequestMapping(value = "/getLiteInfoList", method =GET, produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getLiteInfoListHeader(HttpServletRequest request, Model model) {
        List<LiteInfoModel> lists;
        lists = liteInfoModelMapper.getLiteInfoList();
        return JSONObject.toJSONString(lists);
    }
    @RequestMapping(value = "/getLiteInfo", method = POST, produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getLiteInfoHeader(HttpServletRequest request, Model model,
                             @RequestBody JSONObject json) {
        System.out.println(json.getString("title"));
        System.out.println(json.getString("author"));
        System.out.println(json.getString("keywords"));
        System.out.println("hello world!");
        List<LiteInfoModel> lists;
        String title = json.getString("title");
        String keywords=json.getString("keywords");//为啥返回的是null而不是空字符串
        String author = json.getString("author");
        String materialMethod = json.getString("materialMethod");
        String mlMethod = json.getString("mlMethod");//为啥返回的是null而不是空字符串
        lists=liteInfoModelMapper.getLiteInfoListByCondition("%"+title+"%",
                "%"+keywords+"%","%"+author+"%","%"
                        +materialMethod+"%","%"+mlMethod+"%");
        return JSONObject.toJSONString(lists);
    }
    @RequestMapping(value = "/getLiteInfoByName", method = POST, produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getLiteInfoByNameHeader(HttpServletRequest request, Model model) {
        /**
        * @Description: 通过文件名获取文献相关描述信息，一个文献pdf对应一个文献信息model
        * @Param: [request, model, json]
        * @return: java.lang.String
        * @Author: wujunming
        * @Date: 2019/4/9
        */
        String file_name = request.getParameter("filename");
        LiteInfoModel liteInfoModel;
        liteInfoModel=liteInfoModelMapper.getLiteInfoByName(file_name);
        return JSONObject.toJSONString(liteInfoModel);
    }
    @RequestMapping(value = "/getWangLiteData", method = RequestMethod.GET)
    //layui测试
    public @ResponseBody
     Map<String,Object> table(){
//        System.out.println("dddddd");
        List<User> userList = new ArrayList<User>();
        for(int i=0;i<=10;i++){
            User user = new User(1,"2222","222");
            userList.add(user);
        }
        Map<String, Object> userResult = new HashMap<String, Object>();
        userResult.put("code",0);
        userResult.put("msg","success");
        userResult.put("count",1000);
        userResult.put("data",userList);
        return userResult;
    }
}
