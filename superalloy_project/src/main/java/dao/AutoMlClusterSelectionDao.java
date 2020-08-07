package dao;

import entity.ClusterModelEntity;
import entity.MetaFeatureEntity;
import entity.RecommendResultEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoOperations;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Repository;

import javax.swing.text.Document;
import java.util.List;

@Repository
public class AutoMlClusterSelectionDao {
    @Autowired
    private MongoOperations mongo;

    //清除已有的该数据的元特征，确保不冗余
    public void MetaFeature_Clean(String username,String filename){
        mongo.remove(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename)),Document.class,"MetaFeature");
    }

    //只进行取出元特征的操作
    public MetaFeatureEntity MetaFeature_getResult(String username, String filename){
        List<MetaFeatureEntity> lists = mongo.find(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename)),MetaFeatureEntity.class,"MetaFeature");
        return lists.get(0);
    }

    //清除已有的推荐结果，确保不冗余
    public void Recommendation_Clean(String username,String filename){
        mongo.remove(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename)),Document.class,"RecommendResult");
    }

    //取出基于用户的推荐结果操作
    public RecommendResultEntity Recommendation_getUserResult(String username, String filename){
        List<RecommendResultEntity> lists = mongo.find(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("type").is("User_Based")),RecommendResultEntity.class,"RecommendResult");
        return lists.get(0);
    }

    //取出基于物品的推荐结果操作
    public RecommendResultEntity Recommendation_getItemResult(String username, String filename){
        List<RecommendResultEntity> lists = mongo.find(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("type").is("Item_Based")),RecommendResultEntity.class,"RecommendResult");
        return lists.get(0);
    }

    //取出基于模型的推荐结果操作
    public RecommendResultEntity Recommendation_getModelResult(String username, String filename){
        List<RecommendResultEntity> lists = mongo.find(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("type").is("Model_Based")),RecommendResultEntity.class,"RecommendResult");
        return lists.get(0);
    }

    //清除已有的RF推荐结果，确保不冗余
    public void Recommendation_RF_Clean(String username,String filename){
        mongo.remove(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("type").is("RF_Based")),Document.class,"RecommendResult");
    }

    //取出基于RF的推荐结果操作
    public RecommendResultEntity Recommendation_getRFResult(String username, String filename){
        List<RecommendResultEntity> lists = mongo.find(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("type").is("RF_Based")),RecommendResultEntity.class,"RecommendResult");
        return lists.get(0);
    }

    //清楚对应用户、文件、算法的聚类模型，用于更新
    public void ClusterModel_Clean(String username,String filename,String alg){
        mongo.remove(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("algorithm_name").is(alg)),Document.class,"ClusterModel");
    }

    //取出对应用户、文件、算法的聚类模型
    public ClusterModelEntity ClusterModel_getResult(String username,String filename,String alg){
        List<ClusterModelEntity> lists = mongo.find(Query.query(Criteria.where("user_name").is(username).
                and("file_name").is(filename).and("algorithm_name").is(alg)),ClusterModelEntity.class,"ClusterModel");
        return lists.get(0);
    }


}
