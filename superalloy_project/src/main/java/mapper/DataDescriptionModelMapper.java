package mapper;

import model.insert.DataDescriptionModel;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface DataDescriptionModelMapper {
    int insert(DataDescriptionModel record);

    int insertSelective(DataDescriptionModel record);
    @Select("select * from data_description where d_data_description_id=#{0}")
    DataDescriptionModel getDataDescriptionByID(Integer description_id);
    @Select("select * from data_description where eCode = #{eCode}")
    DataDescriptionModel getDataDescriptionByECode(String eCode);

    @Select("select * from data_description where keywords like #{0}")
    List<DataDescriptionModel> getDataDescriptionByRecordName(String keywords);

    @Select("select * from data_description where user_eCode = #{0}")
    List<DataDescriptionModel> getDataDescriptionByUser(String userECode);
    @Select("select * from data_description")
    List<DataDescriptionModel> getDataDescriptionList();
    @Select("select * from wanglitedata")
    List<DataDescriptionModel> getWangLiteDataDescriptionList();
    @Select("select * from wanglitedata where keywords like #{0}")
    List<DataDescriptionModel> getWangLiteDataDescriptionByRecordName(String keywords);
}