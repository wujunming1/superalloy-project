package realms;

import entity.SysAccount;
import org.apache.shiro.authc.*;
import org.apache.shiro.authz.AuthorizationInfo;
import org.apache.shiro.authz.SimpleAuthorizationInfo;
import org.apache.shiro.realm.AuthorizingRealm;
import org.apache.shiro.subject.PrincipalCollection;
import org.apache.shiro.util.ByteSource;
import org.springframework.beans.factory.annotation.Autowired;
import service.SysAccountService;

import java.util.List;

/**
 * Created by piki on 2017/10/29.
 */
public class DBRealm extends AuthorizingRealm {
    @Autowired
    SysAccountService sysAccountService;

    /**
     * 身份验证
     */
    protected AuthenticationInfo doGetAuthenticationInfo(AuthenticationToken token) throws AuthenticationException {
        String username = (String) token.getPrincipal();
        SysAccount sysAccount = sysAccountService.getLoginAccount(username);
        if(null == sysAccount){
            throw new UnknownAccountException();//没找到帐号
        }
        return  new SimpleAuthenticationInfo(
                sysAccount.getUsername(), //用户名
                sysAccount.getPassword(), //密码
//                ByteSource.Util.bytes(sysAccount.getUsername()+sysAccount.getSalt()),
                getName() ); //realm name  ;
    }


    protected AuthorizationInfo doGetAuthorizationInfo(PrincipalCollection principals) {
        SysAccount account = sysAccountService.getLoginAccount();
        if (account != null) {
            SimpleAuthorizationInfo info = new SimpleAuthorizationInfo();

                info.addRole("common");
            return info;
        } else {
            return null;
        }
    }

}
