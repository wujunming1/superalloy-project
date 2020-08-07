package mapper;

import entity.SysAccount;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface SysAccountModelMapper {
    int insert(SysAccount record);

    int insertSelective(SysAccount account);

    @Select("select * from user_table where username = #{username}")
    List<SysAccount> getUserByUsername(String username);
    @Select("update user_table set password=#{password} where username=#{username}")
    void setPassWordByUserName(@Param("password")String password,
                               @Param("username")String username);
    @Select("update user_table set address=#{address},"+
           "occupation=#{occupation},"+"email=#{email},"+"contact=#{contact},"+
            "res_domain=#{res_domain},"+"unit=#{res_unit},"+"res_direction=#{res_direction},"+
            "where username=#{username}")
    void updateUserInfoByName( @Param("username")String username, @Param("address")String address,
                               @Param("occupation")String occupation,@Param("email") String email,
                               @Param("contact") String contact,@Param("res_domain") String res_domain,
                               @Param("res_unit") String res_unit,
                               @Param("res_direction") String res_direction);
}