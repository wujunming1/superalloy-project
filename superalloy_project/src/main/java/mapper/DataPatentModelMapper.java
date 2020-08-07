package mapper;

import model.insert.DataPatentModel;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface DataPatentModelMapper {
    int insert(DataPatentModel record);

    int insertSelective(DataPatentModel record);
//    @Select("select * from data_description where eCode = #{eCode}")
//    DataPatentModel getDataDescriptionByECode(String eCode);

    @Select("select * from data_description where record_name like #{0}")
    List<DataPatentModel> getPatentDatabyRecord(String record);
//    @Select("select * from data_description where user_eCode = #{0}")
//    List<DataDescriptionModel> getDataDescriptionByUser(String userECode);
//    @Select("select * from data_description")
    @Select("select * from patent_data")
    List<DataPatentModel> getPatentDataList();


}