package dao;
import entity.ClusterStatusEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoOperations;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Repository;
import org.bson.Document;
import entity.ClusterIntegrationEntity;
import java.util.Date;
import java.text.SimpleDateFormat;
import java.util.List;


@Repository
public class ClusterIntegrationDao {
    @Autowired
    private MongoOperations mongo;
    //    非自动式
//    public void cleanCluster(String username,String fileName){
//        mongo.remove(Query.query(Criteria.where("username").is(username).
//                and("data_name").is(fileName)
//            ), Document.class,"GA_cluster");
//    }

    public ClusterIntegrationEntity getResult(String username,String fileName){
        List<ClusterIntegrationEntity> lists=mongo.find(Query.query(Criteria.where("username").is(username).
                and("data_name").is(fileName)),ClusterIntegrationEntity.class,"GA_cluster");
        return lists.get(0);
    }

    public List<ClusterStatusEntity> getStatusComplete(String username){
        List<ClusterStatusEntity> lists=mongo.find(Query.query(Criteria.where("username").is(username).
                and("status").is(3)
        ),ClusterStatusEntity.class,"cluster_status");
        return lists;
    }
    public List<ClusterStatusEntity> getStatusRun(String username){
        List<ClusterStatusEntity> lists=mongo.find(Query.query(Criteria.where("username").is(username).
                and("status").is(2)
        ),ClusterStatusEntity.class,"cluster_status");
        return lists;
    }
    public List<ClusterStatusEntity> getStatusWait(String username){
        List<ClusterStatusEntity> lists=mongo.find(Query.query(Criteria.where("username").is(username).
                and("status").is(1)
        ),ClusterStatusEntity.class,"cluster_status");
        return lists;
    }

    //    自动式
    public void Auto_cleanCluster(String username,String fileName){
//        mongo.remove(Query.query(Criteria.where("username").is(username).
//                        and("data_name").is(fileName).
//               and("status").is(3)
//        ), Document.class,"cluster_status");
        mongo.remove(Query.query(Criteria.where("username").is(username).
                        and("data_name").is(fileName)
//                and("algorithm").is(Algorithm).
        ), Document.class,"GA_cluster");
    }
    public void Auto_cleanStatus(String username,String fileName){
        mongo.remove(Query.query(Criteria.where("username").is(username).
                        and("data_name").is(fileName)

        ), Document.class,"cluster_status");

    }



}
