package mapper;


import model.insert.PatentDataModel;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface PatentDataModelMapper {
    int insert(PatentDataModel record);

    int insertSelective(PatentDataModel record);
    @Select("Select * from patent_data")
    List<PatentDataModel> getPatentDataList();
    @Select("delete from patent_data where Alloy_Index=#{0}")
    void deletePatent(Integer index);//这里需要修改
    //除了字段限制性参数用#获取相应值，其他的例如字段名用$去获取
    @Select("Select * from patent_data where ${condition}>=#{lowlimit} " +
            "and ${condition}<=#{upperlimit}")
    List<PatentDataModel> getPatentDataByCondition(@Param("condition")String condition,
                                                   @Param("lowlimit")Double lowlimit,
                                                   @Param("upperlimit")Double upplimit);
}