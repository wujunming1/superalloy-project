package mapper;

import model.insert.DataLiteratureModel;

public interface DataLiteratureModelMapper {
    int insert(DataLiteratureModel record);

    int insertSelective(DataLiteratureModel record);
}