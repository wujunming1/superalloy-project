package entity;

import org.springframework.data.annotation.Id;

import java.util.List;

public class RecommendResultEntity {
    @Id
    String id;
    String date;
    String file_name;
    String user_name;
    List<String> alg_recommend;
    String type;

    public String getFile_name() {
        return file_name;
    }

    public void setFile_name(String file_name) {
        this.file_name = file_name;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getUser_name() {
        return user_name;
    }

    public void setUser_name(String user_name) {
        this.user_name = user_name;
    }

    public List<String> getAlg_recommend() {
        return alg_recommend;
    }

    public void setAlg_recommend(List<String> alg_recommend) {
        this.alg_recommend = alg_recommend;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getType() {
        return type;
    }
}
