package entity;

import org.springframework.data.annotation.Id;

import java.util.List;

public class MetaFeatureEntity {
    @Id
    String id;
    String date;
    List<String> mf_list;
    String file_name;
    String user_name;
    List<Double> meta_feature;
    List<String> header;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public List<String> getMf_list() {
        return mf_list;
    }

    public void setMf_list(List<String> mf_list) {
        this.mf_list = mf_list;
    }

    public String getFile_name() {
        return file_name;
    }

    public void setFile_name(String file_name) {
        this.file_name = file_name;
    }

    public String getUser_name() {
        return user_name;
    }

    public void setUser_name(String user_name) {
        this.user_name = user_name;
    }

    public List<Double> getMeta_feature() {
        return meta_feature;
    }

    public void setMeta_feature(List<Double> meta_feature) {
        this.meta_feature = meta_feature;
    }

    public void setHeader(List<String> header) {
        this.header = header;
    }
}
