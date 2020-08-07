package service;

import dao.SysAccountDao;
import entity.SysAccount;
import mapper.SysAccountModelMapper;
import org.apache.shiro.SecurityUtils;
import org.apache.shiro.subject.Subject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Date;

/**
 * Created by piki on 2017/10/29.
 */
@Service
public class SysAccountService {
    @Autowired
    private SysAccountDao sysAccountDao;
    @Autowired
    private SysAccountModelMapper sysAccountModelMapper;

    public SysAccount getLoginAccount(String username){
        return sysAccountDao.findByUsername(username);
    }

    //获取登录用户信息 TODO ThreadLocal
    public SysAccount getLoginAccount(){
        Subject subject = SecurityUtils.getSubject();
        SysAccount sysAccount = getLoginAccount( subject.getPrincipal().toString());
        return sysAccount;
    }

    public SysAccount addAccount(String username,String password){
        if(checkAccountUsernameUnique(username)){
            //唯一
            SysAccount sysAccount = new SysAccount(username,password);
//            sysAccount.setCreateTime(new Date());
//            sysAccount.setStatus(AccountStatus.NORMAL);
            sysAccountDao.addSysAcount(sysAccount);
            return sysAccount;
        }
        return null;
    }

    public boolean checkAccountUsernameUnique(String username){
        SysAccount sysAccount = sysAccountDao.findByUsername(username);
        return sysAccount == null ? true : false;
    }
    public boolean changePassword(String password,String username){
        sysAccountModelMapper.setPassWordByUserName(password,username);
        return true;
    }
}
