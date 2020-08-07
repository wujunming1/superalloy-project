package service;

import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.Random;
import java.util.UUID;

/**
 * Created by piki on 2018/4/2.
 */
@Service
public class ECodeService {

    public String  getECode(String traceECode,int version,String userECode) throws Exception{
        String identify=getIdentify(traceECode);
        if(version>=16){
            throw new Exception("version不为1");
        }
        if(StringUtils.isEmpty(traceECode)){
            traceECode="";
        }
        else if(traceECode.length()!=32){
            throw new Exception("traceECode不是32位");
        }
        if(userECode==null||userECode.length()!=8){
            throw new Exception("userECode不是8位");
        }
        String v=  Integer.toHexString(version);
        String nsi="060";
        Random rand = new Random();
        String md= UUID.randomUUID().toString().replaceAll("-", "").toUpperCase();
        return identify+"453D"+v+nsi+md+traceECode+userECode;
    }

    private String getIdentify(String trace){
        if(!StringUtils.isEmpty(trace)){
            return "E0";
        }
        else{
            return "A0";
        }

    }



}
