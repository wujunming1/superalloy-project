package mapper;


import model.insert.LiteInfoModel;
import model.insert.PatentDataModel;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface LiteInfoModelMapper {
    int deleteByPrimaryKey(Integer id);

    int insert(LiteInfoModel record);

    int insertSelective(LiteInfoModel record);

    LiteInfoModel selectByPrimaryKey(Integer id);

    int updateByPrimaryKeySelective(LiteInfoModel record);

    int updateByPrimaryKey(LiteInfoModel record);
    @Select("select * from data_literature where file_attachment=#{0}")
    LiteInfoModel getLiteInfoByName(String filename);
    @Select("select * from data_literature")
    List<LiteInfoModel> getLiteInfoList();
    @Select("select * from data_literature where titleE like #{0} and keywordsE like #{1} " +
            "and authorE like #{2} and material_method like #{3} and ml_method like #{4}")
    List<LiteInfoModel> getLiteInfoListByCondition(String titleE,
                                                   String keywords,String author,
                                                   String materialMethod,
                                                   String mlMethod);
}