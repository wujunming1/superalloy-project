package dao;

import entity.SysAccount;
import mapper.SysAccountModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoOperations;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Created by piki on 2017/11/5.
 */
@Repository
public class SysAccountDao {
    @Autowired
    private MongoOperations mongo;

    @Autowired
    private SysAccountModelMapper sysAccountModelMapper;

    public SysAccount findByUsername(String username){
        List<SysAccount> users=sysAccountModelMapper.getUserByUsername(username);
        if(users.size()>0){
            return users.get(0);
        }
        else{
            return null;
        }
    }


    public SysAccount addSysAcount(SysAccount account){
        sysAccountModelMapper.insert(account);
        return account;
    }
}
