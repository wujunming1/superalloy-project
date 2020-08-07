package dao;

import entity.RemainFeatureFour;
import entity.RemainFeatureOne;
import entity.RemainFeatureThree;
import entity.RemainFeatureTwo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoOperations;
import org.bson.Document;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Set;

/**
 * Created by piki on 2017/11/13.
 */
@Repository
public class FilterDao {
    @Autowired
    private MongoOperations mongo;

    public void cleanFilter(String username){
        mongo.remove( Query.query(Criteria.where("username").is(username)),Document.class,"outputparameter");

    }

    public RemainFeatureOne getRemainOne(String username){
       List<RemainFeatureOne> lists= mongo.find( Query.query(Criteria.where("username").is(username).and("type").is("onelayeroutput")),RemainFeatureOne.class,"outputparameter");

        return lists.get(0);
    }
    public RemainFeatureTwo getRemainTwo(String username){
        List<RemainFeatureTwo> lists= mongo.find( Query.query(Criteria.where("username").is(username).and("type").is("twolayeroutput")),RemainFeatureTwo.class,"outputparameter");

        return lists.get(0);
    }
    public RemainFeatureThree getRemainThree(String username){
        List<RemainFeatureThree> lists= mongo.find( Query.query(Criteria.where("username").is(username).and("type").is("threelayeroutput")),RemainFeatureThree.class,"outputparameter");

        return lists.get(0);
    }

    public RemainFeatureFour getRemainFour(String username){
        List<RemainFeatureFour> lists= mongo.find( Query.query(Criteria.where("username").is(username).and("type").is("fourlayeroutput")),RemainFeatureFour.class,"outputparameter");

        return lists.get(0);
    }
//    public Document getCoefficient(String type){
//        List<Document> lists=mongo.find(Query.query(Criteria.where("type").is(type)),Document.class,"coefficient");
//
//        return lists.get(0);
//    }

    public boolean getDBIsEmpty(){
        List<Document> lists=mongo.findAll(Document.class,"coefficient");
        if(lists.size()<=0){
            return true;
        }
        lists=mongo.findAll(Document.class,"remainfeature");
        if(lists.size()<=0){
            return true;
        }
        return false;
    }
}
