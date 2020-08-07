package service;

import dao.DataDao;
import entity.SysAccount;
import mapper.DataDescriptionModelMapper;
import mapper.DataPatentModelMapper;
import mapper.PatentDataModelMapper;
import model.insert.DataDescriptionModel;
import model.insert.PatentDataModel;
import org.bson.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.util.List;

/**
 * Created by piki on 2017/9/21.
 */
@Service
public class DataService {
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    DataDao dataDao;
    @Resource
    private DataDescriptionModelMapper dataDescriptionModelMapper;
    @Autowired
    private DataPatentModelMapper dataPatentModelMapper;
    @Autowired
    private PatentDataModelMapper patentDataModelMapper;
    public List getPatentDataRecord(){
        return patentDataModelMapper.getPatentDataList();
    }
//    public List<PatentDataModel> getPatentDataByRecordName(String recordname){
//        return patentDataModelMapper.getPatentDatabyRecord(recordname);
//    }
    public List<String> getUserDataRecord(String username){
       return   dataDao.getDataRecord(username);
    }

    public List<Document> getData(String username, String record){
        return   dataDao.getData(username,record);
    }

    public List<Document> getDataByECode(String eCode){
        return   dataDao.getDataByECode(eCode);
    }

    public List search(String condition){
        List<Document> recordList=dataDao.getSearchRes(condition);
        return recordList;
    }

    public DataDescriptionModel getDataDescription(String record){
        SysAccount account = sysAccountService.getLoginAccount();
        String eCode=dataDao.getECode(account.getUsername(),record);
        return dataDescriptionModelMapper.getDataDescriptionByECode(eCode);
    }

    public DataDescriptionModel getDataDescriptionByECode(String eCode){
        return dataDescriptionModelMapper.getDataDescriptionByECode(eCode);
    }

    public void dataModify(String id,String field,String value){
        dataDao.dataModify(id,field,value);
    }

    public String getRecordByECode(String eCode){
        return dataDao.getRecodeByECode(eCode);
    }
}
