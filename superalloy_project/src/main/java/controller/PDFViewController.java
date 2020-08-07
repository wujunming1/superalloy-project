package controller;

import org.apache.commons.io.IOUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;

import static org.springframework.web.bind.annotation.RequestMethod.POST;


@Controller
public class PDFViewController {

    @RequestMapping("/index")
    public String index(){
        return "index";
    }
    @RequestMapping(value = "/preview", method = RequestMethod.GET)
    public void pdfStreamHandler(HttpServletRequest request, HttpServletResponse response) {

        File file = new File("E:/test.pdf");
        if (file.exists()){
            byte[] data = null;
            try {
                FileInputStream input = new FileInputStream(file);
                data = new byte[input.available()];
                input.read(data);

//                response.reset();
//                response.setContentType("application/octet-stream");
//                response.setCharacterEncoding("utf-8");
//                response.setHeader("Content-Disposition", "attachment;filename=" + "test.pdf");

                response.getOutputStream().write(data);
                input.close();
            } catch (Exception e) {
                System.out.println("pdf文件处理异常：" + e);
            }

        }else{
            return;
        }
    }


//    public String showPdf(HttpServletRequest request, HttpServletResponse response) throws Exception {
//        try {
//            // 网络pdf文件全路径
//            String pdfUrl ="https://hljjp.oss-cn-hangzhou.aliyuncs.com/epdfimg/stu_grade_2017121450829122_185125_20171214142015.pdf";
//            URL url = new URL(pdfUrl);
//            HttpURLConnection conn = (HttpURLConnection)url.openConnection();
//            conn.setConnectTimeout(5*1000);
//            InputStream inputStream = conn.getInputStream();
//            response.setHeader("Content-Disposition", "attachment;fileName=结业.pdf");
//            response.setContentType("multipart/form-data");
//            OutputStream outputStream = response.getOutputStream();
//            IOUtils.write(IOUtils.toByteArray(inputStream), outputStream);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//        return null;
//    }
}