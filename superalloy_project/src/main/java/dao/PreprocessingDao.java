package dao;

import entity.PreprocessingEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoOperations;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Repository;

import javax.swing.text.Document;
import java.util.List;

@Repository
public class PreprocessingDao {
    @Autowired
    private MongoOperations mongo;


    //清除已有的alg模型，确保不冗余
    public void AlgModel_Clean(String username,String filename,String alg){
        mongo.remove(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("algorithm_name").is(alg)),Document.class,"PreprocessedData");
    }

    //取出对应的alg模型
    public PreprocessingEntity AlgModel_getResult(String username, String filename, String alg){
        List<PreprocessingEntity> lists = mongo.find(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("algorithm_name").is(alg)),PreprocessingEntity.class,"PreprocessedData");
        return lists.get(0);
    }
}
