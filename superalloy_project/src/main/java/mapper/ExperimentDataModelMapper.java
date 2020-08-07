package mapper;

import model.insert.ExperimentDataModel;

public interface ExperimentDataModelMapper {
    int insert(ExperimentDataModel record);

    int insertSelective(ExperimentDataModel record);
}