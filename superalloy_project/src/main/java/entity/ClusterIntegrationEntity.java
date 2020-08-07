package entity;

import org.springframework.data.annotation.Id;

import java.util.List;
public class ClusterIntegrationEntity {
    @Id
    String id;
    String date;
    String type;
    String username;
    String data_name;
    int features_count;
    List<String> features_names;
    List<List<Double>> data;
    int sample_count;
    Double score;  List<Integer> algorithm;
    List<Integer> feature;
    List<Double> predict_label;
    List<Double> real_label;

    List<List<Double>> xy;

    public void setId(String id) {
        this.id = id;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public void setType(String type) {
        this.type = type;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public void setData_name(String data_name) {
        this.data_name = data_name;
    }

    public void setFeatures_count(int features_count) {
        this.features_count = features_count;
    }

    public void setFeatures_names(List<String> features_names) {
        this.features_names = features_names;
    }

    public void setData(List<List<Double>> data) {
        this.data = data;
    }

    public void setSample_count(int sample_count) {
        this.sample_count = sample_count;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public void setAlgorithm(List<Integer> algorithm) {
        this.algorithm = algorithm;
    }

    public void setFeature(List<Integer> feature) {
        this.feature = feature;
    }

    public void setPredict_label(List<Double> predict_label) {
        this.predict_label = predict_label;
    }

    public void setReal_label(List<Double> real_label) {
        this.real_label = real_label;
    }

    public String getId() {
        return id;
    }

    public String getDate() {
        return date;
    }

    public String getType() {
        return type;
    }

    public String getUsername() {
        return username;
    }

    public String getData_name() {
        return data_name;
    }

    public int getFeatures_count() {
        return features_count;
    }

    public List<String> getFeatures_names() {
        return features_names;
    }

    public List<List<Double>> getData() {
        return data;
    }

    public int getSample_count() {
        return sample_count;
    }

    public Double getScore() {
        return score;
    }

    public List<Integer> getAlgorithm() {
        return algorithm;
    }

    public List<Integer> getFeature() {
        return feature;
    }

    public List<Double> getPredict_label() {
        return predict_label;
    }

    public List<Double> getReal_label() {
        return real_label;
    }

    public List<List<Double>> getXy() {
        return xy;
    }

    public void setXy(List<List<Double>> xy) {
        this.xy = xy;
    }
}
