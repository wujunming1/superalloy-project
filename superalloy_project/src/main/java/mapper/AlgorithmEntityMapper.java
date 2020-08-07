package mapper;

import entity.AlgorithmEntity;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface AlgorithmEntityMapper {
    int insert(AlgorithmEntity algo);

    int insertSelective(AlgorithmEntity algo);

    @Select("select * from algorithm where id = #{id}")
    AlgorithmEntity getAlgoById(String id);
}