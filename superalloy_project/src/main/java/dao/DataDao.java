package dao;

import com.mongodb.MongoClient;
import com.mongodb.client.FindIterable;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoCursor;
import com.mongodb.client.MongoDatabase;
import entity.RemainFeatureOne;
import entity.SysAccount;
import org.bson.Document;
import org.python.antlr.ast.Str;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoOperations;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.data.mongodb.core.query.Update;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;

@Repository
public class DataDao {

	@Autowired
	private MongoOperations mongo;

	// private ReadExcel readExcel=new ReadExcel();

	public ArrayList<ArrayList<Object>> listToDB( ArrayList<ArrayList<Object>> list,String dataRecord,String collect,String eCode){
		try{   
			List<Object> title=list.get(0);
			System.out.println("sdddd:"+title);
			// 连接到 mongodb 服务
	         MongoClient mongoClient = new MongoClient( "localhost" , 27017 );
	         
	         // 连接到数据库
	         MongoDatabase mongoDatabase = mongoClient.getDatabase("MongoDB_Data");
	         System.out.println("Connect to database successfully");
	         
	         MongoCollection<Document> collection = mongoDatabase.getCollection(collect);
	         System.out.println("集合 test 选择成功");
			//插入文档
			/**
			 * 1. 创建文档 org.bson.Document 参数为key-value的格式
			 * 2. 创建文档集合List<Document>
			 * 3. 将文档集合插入数据库集合中 mongoCollection.insertMany(List<Document>) 插入单个文档可以用 mongoCollection.insertOne(Document)
			 * */
	         List<Document> documents = new ArrayList<Document>();
	         for(int i=1;i<list.size();i++){
		         Document document = new Document();
				 document.append("username","piki");
				 document.append("record",dataRecord);
				 document.append("eCode_",eCode);
//				 document.append("")
				 for(int j=0;j<title.size();j++){
					 try{
						 document.append(title.get(j).toString(), list.get(i).get(j));
					 }catch(Exception e){
						 e.printStackTrace();
					 }
				 }
				 documents.add(document);
	         }

	         collection.insertMany(documents);
	         list.get(0).add("_id");
			for(int i=1;i<list.size();i++){
				list.get(i).add(documents.get(i-1).get("_id").toString());
			}
	         System.out.println("文档插入成功");
	         return list;
//			ArrayList<ArrayList<Object>> result=mongo.find( Query.query(Criteria.where("eCode_").is(eCode)), ArrayList<Object>.class,"outputparameter");
	      }catch(Exception e){
	         System.err.println( e.getClass().getName() + ": " + e.getMessage() );
	      }
	      return list;
	}

	public void addDataRecord( String dataRecord ,String user,String eCode){
		try{
			// 连接到 mongodb 服务
			MongoClient mongoClient = new MongoClient( "localhost" , 27017 );

			// 连接到数据库
			MongoDatabase mongoDatabase = mongoClient.getDatabase("MongoDB_Data");

			MongoCollection<Document> collection = mongoDatabase.getCollection("dataRecord");
			//插入文档
			/**
			 * 1. 创建文档 org.bson.Document 参数为key-value的格式
			 * 2. 创建文档集合List<Document>
			 * 3. 将文档集合插入数据库集合中 mongoCollection.insertMany(List<Document>) 插入单个文档可以用 mongoCollection.insertOne(Document)
			 * */
			Document document = new Document();
			document.append("username", user);
			document.append("record", dataRecord);
			document.append("eCode",eCode);
			collection.insertOne(document);
		}catch(Exception e){
			System.err.println( e.getClass().getName() + ": " + e.getMessage() );
		}
	}

	public List<String> getDataRecord(String username){
		List<String> recordList=new ArrayList<String>();

		try{

			// 连接到 mongodb 服务
			MongoClient mongoClient = new MongoClient( "localhost" , 27017 );

			// 连接到数据库
			MongoDatabase mongoDatabase = mongoClient.getDatabase("MongoDB_Data");

			MongoCollection<Document> collection = mongoDatabase.getCollection("dataRecord");
			System.out.println("集合 test 选择成功");

			FindIterable<Document>  findIterable = collection.find();
			MongoCursor<Document> mongoCursor = findIterable.iterator();
			while(mongoCursor.hasNext()) {
				Document document = mongoCursor.next();
				String cursorUser = (String) document.get("username");
				System.out.println(cursorUser);
				if (cursorUser != null && cursorUser.equals(username)) {
					String record = (String) (document.get("record"));
					recordList.add(record);
					System.out.println(record);
				}

			}

		}catch(Exception e){
			System.err.println( e.getClass().getName() + ": " + e.getMessage() );
		}
		return recordList;
	}

	public String getECode(String username,String record){
		String eCode=new String();

		try{

			// 连接到 mongodb 服务
			MongoClient mongoClient = new MongoClient( "localhost" , 27017 );

			// 连接到数据库
			MongoDatabase mongoDatabase = mongoClient.getDatabase("MongoDB_Data");

			MongoCollection<Document> collection = mongoDatabase.getCollection("dataRecord");
			System.out.println("集合 test 选择成功");

			FindIterable<Document>  findIterable = collection.find();
			MongoCursor<Document> mongoCursor = findIterable.iterator();
			while(mongoCursor.hasNext()) {
				Document document = mongoCursor.next();
				String cursorUser = (String) document.get("username");
				String cursorRecord = (String) document.get("record");

				if (cursorUser != null && cursorUser.equals(username)&&cursorRecord!=null&&cursorRecord.equals(record)) {
					return (String) document.get("eCode");
				}

			}

		}catch(Exception e){
			System.err.println( e.getClass().getName() + ": " + e.getMessage() );
		}
		return "";
	}

	public List<Document> getData(String username,String record){
		List<Document> recordList=new ArrayList<Document>();

		try{
			// 连接到 mongodb 服务
			MongoClient mongoClient = new MongoClient( "localhost" , 27017 );

			// 连接到数据库
			MongoDatabase mongoDatabase = mongoClient.getDatabase("MongoDB_Data");

			MongoCollection<Document> collection = mongoDatabase.getCollection("dataUpload");
			System.out.println("集合 test 选择成功");

			FindIterable<Document>  findIterable = collection.find();
			MongoCursor<Document> mongoCursor = findIterable.iterator();
			while(mongoCursor.hasNext()){
				Document document=mongoCursor.next();
				String cursorUser=(String)document.get("username");
				String cursorRecord=(String)document.get("record");
				if(cursorUser!=null&&cursorRecord!=null&&cursorUser.equals(username)&&cursorRecord.equals(record)){ //文件名匹配有问题！！！！！！
					recordList.add(document);
				}
			}

		}catch(Exception e){
			System.err.println( e.getClass().getName() + ": " + e.getMessage() );
		}
		return recordList;
	}

	public List<Document> getDataByECode(String eCode){
		List<Document> recordList=new ArrayList<Document>();

		try{
			// 连接到 mongodb 服务
			MongoClient mongoClient = new MongoClient( "localhost" , 27017 );

			// 连接到数据库
			MongoDatabase mongoDatabase = mongoClient.getDatabase("MongoDB_Data");

			MongoCollection<Document> collection = mongoDatabase.getCollection("dataUpload");
			System.out.println("集合 test 选择成功");

			FindIterable<Document>  findIterable = collection.find();
			MongoCursor<Document> mongoCursor = findIterable.iterator();
			while(mongoCursor.hasNext()){
				Document document=mongoCursor.next();
				String cursorEcode=(String)document.get("eCode_");
				if(cursorEcode!=null&&cursorEcode.equals(eCode)){ //文件名匹配有问题！！！！！！
					recordList.add(document);
				}
			}

		}catch(Exception e){
			System.err.println( e.getClass().getName() + ": " + e.getMessage() );
		}
		return recordList;
	}

	public List<String> getDataHeader(String username,String record){
		List<String> header=new ArrayList<String>();
		List<Document> documents=mongo.find(Query.query(Criteria.where("username").is(username).and("record").is(record)),Document.class,"dataUpload");
		if(documents.size()>0){
			header.addAll(documents.get(0).keySet());
		}
		return header;
	}

	public List<String> getDataHeaderByECode(String eCode){
		List<String> header=new ArrayList<String>();
		List<Document> documents=mongo.find(Query.query(Criteria.where("eCode_").is(eCode)),Document.class,"dataUpload");
		if(documents.size()>0){
			header.addAll(documents.get(0).keySet());
		}
		return header;
	}

	public String getRecodeByECode(String eCode){
		List<Document> documents=mongo.find(Query.query(Criteria.where("eCode_").is(eCode)),Document.class,"dataUpload");
		if(documents.size()>0){
			return documents.get(0).getString("record");
		}
		return "";
	}

	//实现home页面中的搜索功能
	public List<Document> getSearchRes(String key) {
		ArrayList<Document> recordList = new ArrayList<Document>();
		try {
			// 连接到 mongodb 服务
			MongoClient mongoClient = new MongoClient("localhost", 27017);
			// 连接到数据库
			MongoDatabase mongoDatabase = mongoClient.getDatabase("MongoDB_Data");

			MongoCollection<Document> collection = mongoDatabase.getCollection("cif");//需要在mongodb数据库中创建一个search表用来存放所有化合物相应的一些属性
			FindIterable<Document> findIterable = collection.find();
			MongoCursor<Document> mongoCursor = findIterable.iterator();
			while (mongoCursor.hasNext()) {
				Document document = mongoCursor.next();
				Document formula = (Document) document.get("Chemical_formula");
				String txt=(String)formula.get("_chemical_formula_structural");
				if (key.contains(" | ")) {
					String[] keys = key.split(" | ");
					for (String k : keys) {
						if (txt.contains(k.trim())) {
							recordList.add(document);
							break;
						}
					}
				} else if (key.contains("&")) {
					String[] keys = key.split("&");
					boolean passed = true;
					for (String k : keys) {
						if (!txt.contains(k.trim())) {
							passed = false;
							break;
						}
					}
					if (passed) {
						recordList.add(document);
					}
				} else {
					if (txt.contains(key.trim())) {
						recordList.add(document);
					}
				}
			}

		} catch (Exception e) {
			System.err.println(e.getClass().getName() + ": " + e.getMessage());
		}
		return recordList;
	}

	public void dataModify(String id,String field,String value){
		mongo.updateFirst(new Query(Criteria.where( "_id" ).is( id)),
				Update.update( field, value ),"dataUpload");
	}
}
