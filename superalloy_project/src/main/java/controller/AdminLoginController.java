package controller;

import com.alibaba.fastjson.JSONObject;
import entity.SysAccount;
import mapper.SysAccountModelMapper;
import org.apache.shiro.SecurityUtils;
import org.apache.shiro.authc.IncorrectCredentialsException;
import org.apache.shiro.authc.UnknownAccountException;
import org.apache.shiro.authc.UsernamePasswordToken;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import service.SysAccountService;

import javax.annotation.Resource;
import javax.servlet.http.HttpServletRequest;

import java.util.HashMap;
import java.util.Map;

import static org.springframework.web.bind.annotation.RequestMethod.POST;

/**
 * Created by Administrator on 2016/12/8 0008.
 */
@Controller
@RequestMapping("/admin/")
public class AdminLoginController {
    @Autowired
    private SysAccountService sysAccountService;
    @Autowired
    private service.NonAutoMlClusterSelectionService NonAutoMlClusterSelectionService;
    @Autowired
    private SysAccountModelMapper sysAccountModelMapper;
    @RequestMapping(value = "login",method = POST)
    public String login(String username,String password,String password1,Model model){
        UsernamePasswordToken token = new UsernamePasswordToken(username,password);
        String error = null;
//        if(null != rememberMe){
//            token.setRememberMe(true);
//        }
        if(password1.equals(password)){
            error="两次输入密码不一致!";
        }
        try{
            SecurityUtils.getSubject().login(token);
        }catch (UnknownAccountException e){
            error= "用户名/密码错误";
        }catch (IncorrectCredentialsException e){
            error= "用户名/密码错误";
        }catch (Exception e){
            error= "未知错误";
        }
        if(null != error){
            model.addAttribute("error", error);
            model.addAttribute("username",username);
            return "admin/login";
        }
        return  "redirect:/page/index";
    }

    @RequestMapping(value = "logout",method = RequestMethod.GET)
    public String logout(){
        SecurityUtils.getSubject().logout();
        return  "redirect:/page/index";
    }

    @RequestMapping(value = "getUser",method = RequestMethod.GET,produces = "application/json; charset=utf-8")
    public @ResponseBody String getUser(){
        SysAccount account = sysAccountService.getLoginAccount();
        return JSONObject.toJSONString(account);
    }

    @RequestMapping(value = "register",method = POST)
    public String register(String username,String password,Model model){
        SysAccount account = sysAccountService.addAccount(username,password);
        String error = null;
        if(account==null){
            error= "已有同名用户";
        }
        else{
            try{
                UsernamePasswordToken token = new UsernamePasswordToken(username,password);
                SecurityUtils.getSubject().login(token);
            }catch (Exception e){
                error= "未知错误";
            }
        }
        if(null != error){
            model.addAttribute("error", error);
            model.addAttribute("username",username);
            return "admin/register";
        }
        return "redirect:/page/index";

    }
    @RequestMapping(value = "changePassword",method = POST)
    public String passwordChange(String username,String password,
                                 String newpassword, String repassword, Model model){
        String error = null;
        username = "piki";
        if(!newpassword.equals(repassword)){
            error = "两次输入密码不一致!";
        }else{
            sysAccountService.changePassword(newpassword,username);
        }
        if(error!=null){
            model.addAttribute("error",error);
        }
        return "redirect:/page/loginPage";

    }
    @RequestMapping(value = "/updateUserInfo", method = POST,
            produces = "application/json; charset=utf-8")
    public @ResponseBody
    String getLiteInfoHeader(HttpServletRequest request, Model model,
                             @RequestBody JSONObject json) {
        Map<String,String> stringMap = new HashMap<String, String>();
        stringMap.put("状态码","200");
        System.out.println(200+stringMap.get("状态码"));
        String username = json.getString("username");
        String age=json.getString("age");//为啥返回的是null而不是空字符串
        String occupation = json.getString("occupation");
        String email = json.getString("email");
        String phone = json.getString("phone");//为啥返回的是null而不是空字符串
        String res_domain = json.getString("res_domain");
        String res_unit = json.getString("res_unit");
        String res_direction = json.getString("res_direction");
        String address = json.getString("address");
        sysAccountModelMapper.updateUserInfoByName(username,address,
                occupation,email,phone,res_domain,res_unit,res_direction);
        return JSONObject.toJSONString(stringMap);
    }

    @RequestMapping("/getModel")
    public @ResponseBody
//    username:用户名,
    String Model(HttpServletRequest request, Model model,
                 @RequestParam("username") String username){
        String json = NonAutoMlClusterSelectionService.Model(username);
        System.out.println(json);
        return json;
    }
}
