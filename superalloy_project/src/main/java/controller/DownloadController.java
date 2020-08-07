package controller;

import com.alibaba.fastjson.JSONObject;
import dao.DataDao;
import entity.AlgorithmEntity;
import entity.DataModifyEntity;
import entity.SysAccount;
import mapper.*;
import model.insert.*;
import org.apache.commons.io.FileUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import python.ReadCifService;
import service.*;

import javax.annotation.Resource;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;
import java.net.URLEncoder;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import static org.springframework.web.bind.annotation.RequestMethod.POST;

/**
 * Created by piki on 2018/1/28.
 */


@Controller
@RequestMapping("/file")
public class DownloadController {

    @Value("E:\\")
    private String downloadPath;

    @RequestMapping(value ="/sampledata",method = POST)
    public void sampledata(HttpServletRequest req, HttpServletResponse resp) {
        String filename = req.getParameter("filename");
        DataInputStream in = null;
        OutputStream out = null;
        try{
            resp.reset();// 清空输出流
            String resultFileName = filename + System.currentTimeMillis() + ".xlsx";
            resultFileName = URLEncoder.encode(resultFileName,"UTF-8");
            resp.setCharacterEncoding("UTF-8");
            resp.setHeader("Content-disposition", "attachment; filename=" + resultFileName);// 设定输出文件头
            resp.setContentType("application/msexcel");// 定义输出类型
            //输入流：本地文件路径
            in = new DataInputStream(
                    new FileInputStream(new File(downloadPath + "test.xlsx")));
            //输出流
            out = resp.getOutputStream();
            //输出文件
            int bytes = 0;
            byte[] bufferOut = new byte[1024];
            while ((bytes = in.read(bufferOut)) != -1) {
                out.write(bufferOut, 0, bytes);
            }
        } catch(Exception e){
            e.printStackTrace();
            resp.reset();
            try {
                OutputStreamWriter writer = new OutputStreamWriter(resp.getOutputStream(), "UTF-8");
                String data = "<script language='javascript'>alert(\"\\u64cd\\u4f5c\\u5f02\\u5e38\\uff01\");</script>";
                writer.write(data);
                writer.close();
            } catch (IOException e1) {
                e1.printStackTrace();
            }
        }finally {
            if(null != in) {
                try {
                    in.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if(null != out) {
                try {
                    out.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

    }
    @RequestMapping(value="/download",method= RequestMethod.GET) //匹配的是href中的download请求
    public ResponseEntity<byte[]> download(HttpServletRequest request, @RequestParam("filename") String filename,
                                           Model model, HttpServletResponse response) throws IOException{
        /**
        * @Description:  单个/批量下载excel表文件
        * @Param: [request, filename, model, response]
        * @return: org.springframework.http.ResponseEntity<byte[]>
        * @Author: wujunming
        * @Date: 2019/4/9
        */
        String downloadFilePath="D:\\filedownload\\";//从我们的上传文件夹中去取
        System.out.println("文件名"+filename);
        String name="";
        request.setCharacterEncoding("UTF-8");
        //第一步：设置响应类型
        response.setContentType("application/force-download");//应用程序强制下载
        //第二读取文件
        //String path ="E:\\研究生\\项目\\本科项目\\18年本科毕业设计\\店铺聚类\\2-28平台修改\\平台\\dlg-security\\dlg-web\\src\\main\\webapp\\statics\\file\\cluster\\sale\\clother\\杭州全数据表单.xlsx" ;
        InputStream in = new FileInputStream(downloadFilePath+filename);
        //设置响应头，对文件进行url编码
        name = URLEncoder.encode(filename, "UTF-8");
        response.setHeader("Content-Disposition", "attachment;filename="+name);
        response.setContentLength(in.available());
        //第三步：老套路，开始copy
        OutputStream out = response.getOutputStream();
        byte[] b = new byte[1024];
        int len = 0;
        while((len = in.read(b))!=-1){
            out.write(b, 0, len);
        }
        out.flush();
        out.close();
        in.close();
        return null;
    }
    @RequestMapping(value="/downloadPdf",method= RequestMethod.GET) //匹配的是href中的download请求
    public ResponseEntity<byte[]> downloadPdf(HttpServletRequest request, Model model, HttpServletResponse response) throws IOException{
        /**
        * @Description: 下载pdf文件;
         * pdf的批量下载实现其实是前端获取需下载的文件列表，然后循环地逐一下载
        * @Param: [request, model, response]
        * @return: org.springframework.http.ResponseEntity<byte[]>
        * @Author: wujunming
        * @Date: 2019/4/9
        */
        File file = new File("D:\\filedownload\\"+request.getParameter("filename"));//新建一个文件

        HttpHeaders headers = new HttpHeaders();//http头信息

//        String downloadFileName = new String(filename.getBytes("UTF-8"),"iso-8859-1");//设置编码
        String downloadFileName = request.getParameter("filename");
        headers.setContentDispositionFormData("attachment", downloadFileName);

        headers.setContentType(MediaType.APPLICATION_OCTET_STREAM);

        //MediaType:互联网媒介类型  contentType：具体请求中的媒体类型信息
        System.out.println("文件下载成功!");

        return new ResponseEntity<byte[]>(FileUtils.readFileToByteArray(file),headers, HttpStatus.CREATED);

    }
}
