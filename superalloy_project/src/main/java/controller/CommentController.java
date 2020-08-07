package controller;

import com.alibaba.fastjson.JSONObject;
import dao.DataDao;
import entity.CommentEntity;
import entity.SysAccount;
import mapper.*;
import model.insert.CalculationDataModel;
import model.insert.DataDescriptionModel;
import model.insert.DataLiteratureModel;
import model.insert.ExperimentDataModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;
import python.ReadCifService;
import service.*;

import javax.annotation.Resource;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

import static org.springframework.web.bind.annotation.RequestMethod.POST;

/**
 * Created by piki on 2018/1/28.
 */


@Controller
@RequestMapping("/comment")
public class CommentController {
    @Resource
    private CommentEntityMapper commentEntityMapper;
    @Autowired
    private DataDao dataDao;
    @Autowired
    private SysAccountService sysAccountService;
    @RequestMapping(value ="/comment",method = POST)
    public @ResponseBody
    String comment(CommentEntity comment, String record) {
        SysAccount account = sysAccountService.getLoginAccount();
        String eCode= dataDao.getECode(account.getUsername(),record);
        comment.seteCode(eCode);
        commentEntityMapper.insertSelective(comment);
        return "success";
    }
    
}
