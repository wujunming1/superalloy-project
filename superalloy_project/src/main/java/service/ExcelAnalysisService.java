package service;

/**
 * Created by piki on 2017/9/17.
 */

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;

import org.apache.poi.hssf.usermodel.*;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.ss.usermodel.CellValue;
import org.apache.poi.xssf.usermodel.*;
import org.springframework.stereotype.Service;

@Service
public class ExcelAnalysisService {



    private static DecimalFormat df = new DecimalFormat("0");
    // 默认单元格格式化日期字符串
    private static SimpleDateFormat sdf = new SimpleDateFormat(  "yyyy-MM-dd HH:mm:ss");
    // 格式化数字
    private static DecimalFormat nf = new DecimalFormat("0.00");
    

    public  ArrayList<ArrayList<Object>> readExcel(String path){

        if(path == null){
            return null;
        }
        File file = new File(path);
        if(file.getName().endsWith("xlsx")){
            //处理ecxel2007
//            System.out.println("sddddddddddd");
            return readExcel2007(file);
        }else{
            //处理ecxel2003
            return readExcel2003(file);
        }
    }
    
    /*
     * @return 将返回结果存储在ArrayList内，存储结构与二位数组类似
     * lists.get(0).get(0)表示过去Excel中0行0列单元格
     */
    public  ArrayList<ArrayList<Object>> readExcel2003(File file){
        try{
            ArrayList<ArrayList<Object>> rowList = new ArrayList<ArrayList<Object>>();
            ArrayList<Object> colList;
//            HSSFFormulaEvaluator evaluator= new HSSFFormulaEvaluator();  ;
            HSSFWorkbook wb = new HSSFWorkbook(new FileInputStream(file));
            HSSFFormulaEvaluator evaluator = new HSSFFormulaEvaluator(wb);

            HSSFSheet sheet = wb.getSheetAt(0);
            HSSFRow row;
            HSSFCell cell;
            Object value;
            for(int i = sheet.getFirstRowNum() , rowCount = 0; rowCount < sheet.getPhysicalNumberOfRows() ; i++ ) {
                row = sheet.getRow(i);
                colList = new ArrayList<Object>();
                if (row == null) {
                    //当读取行为空时
                    if (i != sheet.getPhysicalNumberOfRows()) {//判断是否是最后一行
                        rowList.add(colList);
                    }
                    continue;
                } else {
                    rowCount++;
                }
                for (int j = row.getFirstCellNum(); j <= row.getLastCellNum(); j++) {
                    cell = row.getCell(j);
                    if (cell == null || cell.getCellTypeEnum() == CellType.BLANK) {
                        //当该单元格为空
                        if (j != row.getLastCellNum()) {//判断是否是该行中最后一个单元格
                            colList.add("");
                        }
                        continue;
                    }
                    switch (cell.getCellTypeEnum()) {
                        case STRING:
//                            System.out.println(i + "行" + j + " 列 is String type");
                            value = cell.getStringCellValue();
                            break;
                        case NUMERIC:
                            if ("@".equals(cell.getCellStyle().getDataFormatString())) {
                                value = df.format(cell.getNumericCellValue());
                            } else if ("General".equals(cell.getCellStyle()
                                    .getDataFormatString())) {
                                value = nf.format(cell.getNumericCellValue());
                            } else {
                                value = sdf.format(HSSFDateUtil.getJavaDate(cell
                                        .getNumericCellValue()));
                            }
//                            System.out.println(i + "行" + j
//                                    + " 列 is Number type ; DateFormt:"
//                                    + value.toString());
                            break;
                        case BOOLEAN:
//                            System.out.println(i + "行" + j + " 列 is Boolean type");
                            value = Boolean.valueOf(cell.getBooleanCellValue());
                            break;
                        case BLANK:
//                            System.out.println(i + "行" + j + " 列 is Blank type");
                            value = "";
                            break;
                        case FORMULA:
                            System.out.println("公式");
//                            cell.setCellFormula(cell.toString());
                            value=evaluator.evaluateInCell(cell).toString();
                        default:
//                            System.out.println(i + "行" + j + " 列 is default type");
                            value = cell.toString();
                    }// end switch
                    colList.add(value);
                }//end for j
                rowList.add(colList);
            }//end for i

            return rowList;
        }catch(Exception e){
            return null;
        }
    }

    public  ArrayList<ArrayList<Object>> readExcel2007(File file){
        try{
            ArrayList<ArrayList<Object>> rowList = new ArrayList<ArrayList<Object>>();
            ArrayList<Object> colList;
            XSSFWorkbook wb = new XSSFWorkbook(new FileInputStream(file));
            XSSFFormulaEvaluator evaluator = new XSSFFormulaEvaluator(wb);
            XSSFSheet sheet = wb.getSheetAt(0);
            XSSFRow row;
            XSSFCell cell;
            Object value;
            for(int i = sheet.getFirstRowNum() , rowCount = 0; rowCount < sheet.getPhysicalNumberOfRows() ; i++ ){
                row = sheet.getRow(i);
                colList = new ArrayList<Object>();
                if(row == null){
                    //当读取行为空时
                    if(i != sheet.getPhysicalNumberOfRows()){//判断是否是最后一行
                        rowList.add(colList);
                    }
                    continue;
                }else{
                    rowCount++;
                }
                for( int j = row.getFirstCellNum() ; j <= row.getLastCellNum() ;j++){
                    cell = row.getCell(j);
                    if(cell == null || cell.getCellTypeEnum() == CellType.BLANK){
                        //当该单元格为空
                        if(j != row.getLastCellNum()){//判断是否是该行中最后一个单元格
                            colList.add("");
                        }
                        continue;
                    }
//                        System.out.println("公式:"+cell.getCellTypeEnum());

                    switch(cell.getCellTypeEnum()){
                        case STRING:
//                            System.out.println(i + "行" + j + " 列 is String type");
                            value = cell.getStringCellValue();
                            break;
                        case NUMERIC:
                            if ("@".equals(cell.getCellStyle().getDataFormatString())) {
                                value = df.format(cell.getNumericCellValue());
                            } else if ("General".equals(cell.getCellStyle()
                                    .getDataFormatString())) {
                                value = nf.format(cell.getNumericCellValue());
                            } else {
                                value = sdf.format(HSSFDateUtil.getJavaDate(cell
                                        .getNumericCellValue()));
                            }
//                            System.out.println(i + "行" + j
//                                    + " 列 is Number type ; DateFormt:"
//                                    + value.toString());
                            break;
                        case BOOLEAN:
//                            System.out.println(i + "行" + j + " 列 is Boolean type");
                            value = Boolean.valueOf(cell.getBooleanCellValue());
                            break;
                        case BLANK:
//                            System.out.println(i + "行" + j + " 列 is Blank type");
                            value = "";
                            break;
                        case FORMULA:
                            System.out.println("公式");
//                            cell.setCellFormula(cell.toString());
                            value=evaluator.evaluateInCell(cell).toString();
                        default:
//                            System.out.println(i + "行" + j + " 列 is default type");
                            value = cell.toString();
                    }// end switch
                    colList.add(value);
                }//end for j
                rowList.add(colList);
                System.out.println("1111"+rowList);
            }//end for i

            return rowList;
        }catch(Exception e){
            System.out.println("222222");
            System.out.println("exception");
            return null;
        }
    }

    public static void writeExcel(ArrayList<ArrayList<Object>> result,String path){
        if(result == null){
            return;
        }
        HSSFWorkbook wb = new HSSFWorkbook();
        HSSFSheet sheet = wb.createSheet("sheet1");
        for(int i = 0 ;i < result.size() ; i++){
            HSSFRow row = sheet.createRow(i);
            if(result.get(i) != null){
                for(int j = 0; j < result.get(i).size() ; j ++){
                    HSSFCell cell = row.createCell(j);
                    cell.setCellValue(result.get(i).get(j).toString());
                }
            }
        }
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        try
        {
            wb.write(os);
        } catch (IOException e){
            e.printStackTrace();
        }
        byte[] content = os.toByteArray();
        File file = new File(path);//Excel文件生成后存储的位置。
        OutputStream fos  = null;
        try
        {
            fos = new FileOutputStream(file);
            fos.write(content);
            os.close();
            fos.close();
        }catch (Exception e){
            e.printStackTrace();
        }
    }

}
