package mapper;

import entity.CommentEntity;
import entity.SysAccount;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface CommentEntityMapper {
    int insert(CommentEntity comment);

    int insertSelective(CommentEntity comment);

//    @Select("select * from sys_account where username = #{username}")
//    List<SysAccount> getUserByUsername(String username);
}