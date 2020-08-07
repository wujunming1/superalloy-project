package service;

import org.springframework.stereotype.Service;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by piki on 2018/3/9.
 */
@Service
public class CsvAnalysisService {
    
    public ArrayList<ArrayList<Object>> readCsv(String serverFile) {
        File csv = new File(serverFile);  // CSV文件路径
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(csv));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        String line = "";
        String everyLine = "";
        ArrayList<ArrayList<Object>> result = new ArrayList<ArrayList<Object>>();
        try {
            while ((line = br.readLine()) != null)  //读取到的内容给line变量
            {
                ArrayList<Object> listTemp = new ArrayList<Object>();
                listTemp.addAll(Arrays.asList(line.split(",")));
                result.add(listTemp);
            }
            return result;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }
}
