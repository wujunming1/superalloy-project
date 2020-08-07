package controller;

import dao.DataDao;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;
import service.DataService;
import service.ExcelAnalysisService;

import javax.servlet.http.HttpServletRequest;

import static org.springframework.web.bind.annotation.RequestMethod.POST;

/**
 * Created by piki on 2017/10/24.
 */

@Controller
@RequestMapping("/literature")
public class Literature {
    @Autowired
    private ExcelAnalysisService excelAnalysisService;
    @Autowired
    private DataDao dataDao;
    @Autowired
    private DataService dataService;

    @RequestMapping(value ="/excel",method = POST)
    public @ResponseBody
    String uploadLiterature(HttpServletRequest request, Model model,
                             @RequestParam("file") MultipartFile[] files) {
        return null;
    }
}
