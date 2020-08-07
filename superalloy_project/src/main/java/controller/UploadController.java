package controller;

import com.alibaba.fastjson.JSONObject;
import dao.DataDao;
import entity.AlgorithmEntity;
import entity.DataModifyEntity;
import entity.SysAccount;
import mapper.*;
import model.insert.*;
import org.bson.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
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
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import static org.springframework.web.bind.annotation.RequestMethod.POST;

/**
 * Created by piki on 2018/1/28.
 */


@Controller
@RequestMapping("/upload")
public class UploadController {
    @Value("E:\\superlloy\\patent\\")
    private String patentPath;
    @Value("${file.literature}")
    private String literaturePath;
    @Value("${file.data}")
    private String dataPath;
    @Value("${file.calcData}")
    private String calcDataPath;
    @Value("${file.algorithm}")
    private String algoPath;

    @Resource
    private DataPatentModelMapper dataPatentModelMapper;
    @Resource
    private DataLiteratureModelMapper dataLiteratureModelMapper;
    @Resource
    private CalculationDataModelMapper calculationDataModelMapper;
    @Resource
    private DataDescriptionModelMapper dataDescriptionModelMapper;
    @Resource
    private ExperimentDataModelMapper experimentDataModelMapper;
    @Resource
    private AlgorithmEntityMapper algorithmEntityMapper;
    @Autowired
    private DataDao dataDao;
    @Autowired
    private DataService dataService;
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private ExcelAnalysisService excelAnalysisService;
    @Autowired
    private ReadCifService readCifService;
    @Autowired
    private CsvAnalysisService csvAnalysisService;
    @Autowired
    private DataQualityService dataQualityService;
    @Autowired
    private ECodeService eCodeService;
    
//    @RequestMapping(value ="/originData",method = POST)
//    public @ResponseBody
//    String uploadFileHandler() {
//        ArrayList<ArrayList<Object>> result=excelAnalysisSercive.readExcel("E:\\project\\superalloy\\data\\专利数据更新1.xlsx");
//        dataDao.excelListToDB(result,"piki");
//        return null;
//    }

    @RequestMapping(value ="/dataPatent",method = POST)
    public @ResponseBody
    String dataPatent(DataPatentModel patent, @RequestParam MultipartFile[] lfile) {
        SysAccount account = sysAccountService.getLoginAccount();

        String uploadRootPath = patentPath+account.getUsername();
        System.out.println("uploadRootPath=" + uploadRootPath);
        File uploadRootDir = new File(uploadRootPath);
        //
        // Create directory if it not exists.
        if (!uploadRootDir.exists()) {
            uploadRootDir.mkdirs();
        }
        //
        List<File> uploadedFiles = new ArrayList<File>();
        for (int i = 0; i < lfile.length; i++) {
            MultipartFile file = lfile[i];

            // Client File Name
            String name = file.getOriginalFilename();
            System.out.println("Client File Name = " + name);

            if (name != null && name.length() > 0) {
                try {
                    byte[] bytes = file.getBytes();

                    // Create the file on server
                    File serverFile = new File(uploadRootDir.getAbsolutePath()
                            + File.separator + name);

                    // Stream to write data to file in server.
                    BufferedOutputStream stream = new BufferedOutputStream(
                            new FileOutputStream(serverFile));
                    stream.write(bytes);
                    stream.close();
                    //
                    uploadedFiles.add(serverFile);
                    System.out.println("Write file: " + serverFile);
                    patent.setFileAttachment(serverFile.toString());

//                    dataDao.excelListToDB(excelAnalysisSercive.readExcel(serverFile.toString()),name);//解析excel后写入mongodb
//                    dataDao.addDataRecord(name,account.getUsername());
//                    return name;
                } catch (Exception e) {
                    System.out.println("Error Write file: " + name);
                }
            }
        }
        dataPatentModelMapper.insert(patent);

        return "success";
    }

    @RequestMapping(value ="/dataLiterature",method = POST)
    public @ResponseBody
    String dataLiterature( DataLiteratureModel literature,@RequestParam MultipartFile[] lfile) {
        SysAccount account = sysAccountService.getLoginAccount();

        String uploadRootPath = literaturePath+account.getUsername();
        System.out.println("uploadRootPath=" + uploadRootPath);
        File uploadRootDir = new File(uploadRootPath);
        //
        // Create directory if it not exists.
        if (!uploadRootDir.exists()) {
            uploadRootDir.mkdirs();
        }
        //
        List<File> uploadedFiles = new ArrayList<File>();
        for (int i = 0; i < lfile.length; i++) {
            MultipartFile file = lfile[i];

            // Client File Name
            String name = file.getOriginalFilename();
            System.out.println("Client File Name = " + name);

            if (name != null && name.length() > 0) {
                try {
                    byte[] bytes = file.getBytes();

                    // Create the file on server
                    File serverFile = new File(uploadRootDir.getAbsolutePath()
                            + File.separator + name);

                    // Stream to write data to file in server.
                    BufferedOutputStream stream = new BufferedOutputStream(
                            new FileOutputStream(serverFile));
                    stream.write(bytes);
                    stream.close();
                    //
                    uploadedFiles.add(serverFile);
                    System.out.println("Write file: " + serverFile);
                    literature.setFileAttachment(serverFile.toString());

//                    dataDao.excelListToDB(excelAnalysisSercive.readExcel(serverFile.toString()),name);//解析excel后写入mongodb
//                    dataDao.addDataRecord(name,account.getUsername());
//                    return name;
                } catch (Exception e) {
                    System.out.println("Error Write file: " + name);
                }
            }
        }
        dataLiteratureModelMapper.insert(literature);
        
        return "success";
    }

    @RequestMapping(value ="/dataAlgo",method = POST)
    public @ResponseBody
    String dataAlgo(AlgorithmEntity algoEntity, @RequestParam MultipartFile[] lfile) {
        SysAccount account = sysAccountService.getLoginAccount();

        String uploadRootPath = algoPath+account.getUsername();
        System.out.println("uploadRootPath=" + uploadRootPath);
        File uploadRootDir = new File(uploadRootPath);
        //
        // Create directory if it not exists.
        if (!uploadRootDir.exists()) {
            uploadRootDir.mkdirs();
        }
        //
        List<File> uploadedFiles = new ArrayList<File>();
        for (int i = 0; i < lfile.length; i++) {
            MultipartFile file = lfile[i];

            // Client File Name
            String name = file.getOriginalFilename();
            System.out.println("Client File Name = " + name);

            if (name != null && name.length() > 0) {
                try {
                    byte[] bytes = file.getBytes();

                    // Create the file on server
                    File serverFile = new File(uploadRootDir.getAbsolutePath()
                            + File.separator + name);

                    // Stream to write data to file in server.
                    BufferedOutputStream stream = new BufferedOutputStream(
                            new FileOutputStream(serverFile));
                    stream.write(bytes);
                    stream.close();
                    //
                    uploadedFiles.add(serverFile);
                    System.out.println("Write file: " + serverFile);
                    algoEntity.setAlgoLocation(serverFile.toString());

//                    dataDao.excelListToDB(excelAnalysisSercive.readExcel(serverFile.toString()),name);//解析excel后写入mongodb
//                    dataDao.addDataRecord(name,account.getUsername());
//                    return name;
                } catch (Exception e) {
                    System.out.println("Error Write file: " + name);
                }
            }
        }
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
        algoEntity.setAlgoDate(sdf.format(new Date()));
        algorithmEntityMapper.insert(algoEntity);

        return "success";
    }

    @RequestMapping(value ="/dataExperiment",method = POST)
    public @ResponseBody
    String dataExperiment(DataDescriptionModel description, ExperimentDataModel experimentData, @RequestParam MultipartFile[] lfile) {
        SysAccount account = sysAccountService.getLoginAccount();
        String eCode="";
        try {
            eCode=eCodeService.getECode("",1,account.getUserECode());
            description.seteCode(eCode);
        }catch (Exception e){
            e.printStackTrace();
            return "";
        }
        description.setUserECode(account.getUserECode());
        experimentDataModelMapper.insert(experimentData);
        ArrayList<ArrayList<Object>> result=null;
        String uploadRootPath = dataPath+account.getUsername();
        System.out.println("uploadRootPath=" + uploadRootPath);
        File uploadRootDir = new File(uploadRootPath);
        //
        // Create directory if it not exists.
        if (!uploadRootDir.exists()) {
            uploadRootDir.mkdirs();
        }
        //
        List<File> uploadedFiles = new ArrayList<File>();
//        for (int i = 0; i < lfile.length; i++) {
            MultipartFile file = lfile[0];

            // Client File Name
            String name = file.getOriginalFilename();
            System.out.println("Client File Name = " + name);
            description.setRecordName(name);
            dataDescriptionModelMapper.insert(description);
            if (name != null && name.length() > 0) {
                try {
                    byte[] bytes = file.getBytes();

                    // Create the file on server
                    File serverFile = new File(uploadRootDir.getAbsolutePath()
                            + File.separator + name);

                    // Stream to write data to file in server.
                    BufferedOutputStream stream = new BufferedOutputStream(
                            new FileOutputStream(serverFile));
                    stream.write(bytes);
                    stream.close();
                    uploadedFiles.add(serverFile);
                    System.out.println("Write file: " + serverFile);
                    String[] nameList=name.split("\\.");
                    String nameEnd=nameList[nameList.length-1];
                    ArrayList<ArrayList<Object>> dataQuality=null;
                    ArrayList<ArrayList<Object>>  returnData=null;
                    if(nameEnd.equals("xls")||nameEnd.equals("xlsx")){
                        result= excelAnalysisService.readExcel(serverFile.toString());
                        returnData=dataDao.listToDB(result,name,"dataUpload",eCode);//解析excel后写入mongodb
                        dataQuality=dataQualityService.getDataQuality(serverFile.toString(),account.getUsername());
                        
                    }
                    else if(nameEnd.equals("csv")){
                        result=csvAnalysisService.readCsv(serverFile.toString());
                        returnData=dataDao.listToDB(result,name,"dataUpload",eCode);//解析excel后写入mongodb//csv
                        dataQuality=dataQualityService.getDataQuality(serverFile.toString(),account.getUsername());
                    }
                    else if(nameEnd.equals("cif")){
                        readCifService.readCif(serverFile.toString());
                    }
                    dataDao.addDataRecord(name,account.getUsername(),eCode);
                    JSONObject jsonObject = new JSONObject();
                    
                    jsonObject.put("dataResult",JSONObject.toJSONString(returnData));
                    jsonObject.put("dataQuality",JSONObject.toJSONString(dataQuality));
                    jsonObject.put("eCode",eCode);
                    return jsonObject.toJSONString();
                } catch (Exception e) {
                    System.out.println("Error Write file: " + name);
                }
            }

//        }
        return "success";
    }
    
    @RequestMapping(value ="/dataCalc",method = POST)
    public @ResponseBody
    String dataCalc(DataDescriptionModel description, CalculationDataModel calculation, @RequestParam MultipartFile[] lfile, @RequestParam MultipartFile[] cFile) {
        SysAccount account = sysAccountService.getLoginAccount();
        String eCode="";
        try {
            eCode=eCodeService.getECode("",1,account.getUserECode());
            description.seteCode(eCode);
        }catch (Exception e){
            e.printStackTrace();
            return "";
        }
        description.setUserECode(account.getUserECode());
        calculationDataModelMapper.insert(calculation);
        ArrayList<ArrayList<Object>> result=null;
        JSONObject jsonObject = new JSONObject();

        String uploadRootPath = dataPath+account.getUsername();
        System.out.println("uploadRootPath=" + uploadRootPath);
        String uploadCalcDataPath = calcDataPath+account.getUsername();
        System.out.println("uploadCalcDataPath=" + uploadCalcDataPath);

        File uploadRootDir = new File(uploadRootPath);
        if (!uploadRootDir.exists()) {
            uploadRootDir.mkdirs();
        }
        File uploadCalcDataDir = new File(uploadCalcDataPath);
        if (!uploadCalcDataDir.exists()) {
            uploadCalcDataDir.mkdirs();
        }
        List<File> uploadedFiles = new ArrayList<File>();
        MultipartFile file = lfile[0];

        // Client File Name
        String name = file.getOriginalFilename();
        System.out.println("Client File Name = " + name);
        description.setRecordName(name);
        dataDescriptionModelMapper.insert(description);

        if (name != null && name.length() > 0) {
            try {
                byte[] bytes = file.getBytes();

                // Create the file on server
                File serverFile = new File(uploadRootDir.getAbsolutePath()
                        + File.separator + name);

                // Stream to write data to file in server.
                BufferedOutputStream stream = new BufferedOutputStream(
                        new FileOutputStream(serverFile));
                stream.write(bytes);
                stream.close();
                uploadedFiles.add(serverFile);
                System.out.println("Write file: " + serverFile);
                String[] nameList=name.split("\\.");
                String nameEnd=nameList[nameList.length-1];
                ArrayList<ArrayList<Object>> dataQuality=null;
                ArrayList<ArrayList<Object>>  returnData=null;

                if(nameEnd.equals("xls")||nameEnd.equals("xlsx")){
                    result= excelAnalysisService.readExcel(serverFile.toString());
                    returnData=dataDao.listToDB(result,name,"dataUpload",eCode);//解析excel后写入mongodb
                    dataQuality=dataQualityService.getDataQuality(serverFile.toString(),account.getUsername());

                }
                else if(nameEnd.equals("csv")){
                    result=csvAnalysisService.readCsv(serverFile.toString());
                    returnData=dataDao.listToDB(result,name,"dataUpload",eCode);//解析excel后写入mongodb//csv
                    dataQuality=dataQualityService.getDataQuality(serverFile.toString(),account.getUsername());
                }
                else if(nameEnd.equals("cif")){
                    readCifService.readCif(serverFile.toString());
                }
                dataDao.addDataRecord(name,account.getUsername(),eCode);

                jsonObject.put("dataResult",JSONObject.toJSONString(returnData));
                jsonObject.put("dataQuality",JSONObject.toJSONString(dataQuality));
                jsonObject.put("eCode",eCode);
            } catch (Exception e) {
                System.out.println("Error Write file: " + name);
            }
        }
        for (int i = 0; i < cFile.length; i++) {
            MultipartFile calcFile = cFile[i];

            String cName = calcFile.getOriginalFilename();
            System.out.println("Client File Name = " + cName);

            if (name != null && name.length() > 0) {
                try {
                    byte[] bytes = calcFile.getBytes();

                    // Create the file on server
                    File serverFile = new File(uploadCalcDataDir.getAbsolutePath()
                            + File.separator + cName);

                    // Stream to write data to file in server.
                    BufferedOutputStream stream = new BufferedOutputStream(
                            new FileOutputStream(serverFile));
                    stream.write(bytes);
                    stream.close();
                    //
                    uploadedFiles.add(serverFile);
                    System.out.println("Write file: " + serverFile);

//                    dataDao.excelListToDB(excelAnalysisSercive.readExcel(serverFile.toString()),name);//解析excel后写入mongodb
//                    dataDao.addDataRecord(name,account.getUsername());
//                    return name;
                } catch (Exception e) {
                    System.out.println("Error Write file: " + name);
                }
            }
        }
        return jsonObject.toJSONString();
//        return "success";
    }

    @RequestMapping(value ="/dataModify",method = POST)
    public @ResponseBody
    String dataModify(@RequestBody DataModifyEntity entity) {

        dataService.dataModify(entity.getId(),entity.getField(),entity.getValue());
        return "success";
    }


    
}
