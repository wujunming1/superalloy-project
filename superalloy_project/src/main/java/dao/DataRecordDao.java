package dao;

import entity.DataRecord;
import entity.RemainFeatureOne;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoOperations;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
@Repository
public class DataRecordDao {
    @Autowired
    private MongoOperations mongo;

    public List<DataRecord> getRecordByName(String recordName,String username){
        List<DataRecord> lists= mongo.find( Query.query(Criteria.where("record").regex(recordName).and("username").is(username)),DataRecord.class,"dataRecord");
        return lists;
    }
}
