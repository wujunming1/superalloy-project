package mapper;


import model.insert.User_Upload_DataModel;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface User_Upload_DataModelMapper {
    int deleteByPrimaryKey(Integer dDataDescriptionId);

    int insert(User_Upload_DataModel record);

    int insertSelective(User_Upload_DataModel record);

    User_Upload_DataModel selectByPrimaryKey(Integer dDataDescriptionId);

    int updateByPrimaryKeySelective(User_Upload_DataModel record);
    int updateByPrimaryKey(User_Upload_DataModel record);
    @Select("select * from user_upload_data where d_submission_name=#{username}")
    List<User_Upload_DataModel> getUploadDataInfo(@Param("username")String username);
}