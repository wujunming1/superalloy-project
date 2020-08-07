package mapper;

import model.insert.CalculationDataModel;

public interface CalculationDataModelMapper {
    int insert(CalculationDataModel record);

    int insertSelective(CalculationDataModel record);
}