package dao;

import com.mongodb.Mongo;
import entity.ClusterModelEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoOperations;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Repository;

import javax.swing.text.Document;
import java.util.List;

@Repository
public class NonAutoMlClusterSelectionDao {
    @Autowired
    private MongoOperations mongo;

//    ['dbscan', 'kmeans', 'meanshift', 'agglomerative', 'birch', 'affinity']


    //清除已有的alg模型，确保不冗余
    public void AlgModel_Clean(String username,String filename,String alg){
        mongo.remove(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("algorithm_name").is(alg)),Document.class,"NonAutoClusterModel");
    }

    //取出对应的alg聚类模型
    public ClusterModelEntity AlgModel_getResult(String username,String filename,String alg){
        List<ClusterModelEntity> lists = mongo.find(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("algorithm_name").is(alg)),ClusterModelEntity.class,"NonAutoClusterModel");
        return lists.get(0);
    }
    //取出该用户的所有模型
    public List<ClusterModelEntity> AlgModel_getResultbyUsername(String username){
        List<ClusterModelEntity> lists = mongo.find(Query.query(Criteria.where("user_name").is(username)),
                ClusterModelEntity.class,"NonAutoClusterModel");
        return lists;
    }
}
