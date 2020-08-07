package entity;

import org.springframework.data.annotation.Id;

import java.util.List;

public class ClusterModelEntity {
    @Id
    String id;
    String date;
    String user_name;
    String file_name;
    String algorithm_name;
    List<List<Double>> data;
    List<Double> result_label;
    Double score;
    List<String> feature_names;
    int feature_count;
//    List<ParameterList> parameter_list;
    List<List<Double>> xy;

    public void setDate(String date) {
        this.date = date;
    }

    public String getDate() {
        return date;
    }

    public void setUser_name(String user_name) {
        this.user_name = user_name;
    }

    public String getUser_name() {
        return user_name;
    }

    public void setFile_name(String file_name) {
        this.file_name = file_name;
    }

    public String getFile_name() {
        return file_name;
    }

    public String getAlgorithm_name() {
        return algorithm_name;
    }

    public void setAlgorithm_name(String algorithm_name) {
        this.algorithm_name = algorithm_name;
    }

    public void setData(List<List<Double>> data) {
        this.data = data;
    }

    public List<List<Double>> getData() {
        return data;
    }

    public void setResult_label(List<Double> result_label) {
        this.result_label = result_label;
    }

    public List<Double> getResult_label() {
        return result_label;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public Double getScore() {
        return score;
    }

    public int getFeature_count() {
        return feature_count;
    }

    public void setFeature_count(int feature_count) {
        this.feature_count = feature_count;
    }

    public void setFeature_names(List<String> feature_names) {
        this.feature_names = feature_names;
    }

    public List<String> getFeature_names() {
        return feature_names;
    }

    public void setXy(List<List<Double>> xy) {
        this.xy = xy;
    }

    public List<List<Double>> getXy() {
        return xy;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }
}
