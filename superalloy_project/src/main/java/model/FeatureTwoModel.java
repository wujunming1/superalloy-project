package model;

import entity.RemainFeatureTwo;

import java.util.List;

/**
 * Created by piki on 2018/1/25.
 */
public class FeatureTwoModel {
    String id;
    Double incor_threshold;
    String username;
    List<Integer> result2;
    List<Integer> filter_indices;
    Double ficor_threshold;
    List<Double> corcoefficient;
    String type;
    List<String> header;
    
    
    public FeatureTwoModel(RemainFeatureTwo dao){
        this.id=dao.getId();
        this.incor_threshold=dao.getIncor_threshold();
        this.username=dao.getUsername();
        this.result2=dao.getResult2();
        this.ficor_threshold=dao.getFicor_threshold();
        this.filter_indices=dao.getFilter_indices();
        this.corcoefficient=dao.getCorcoefficent();
        this.type=dao.getType();
        this.header=dao.getHeader();
    }
    
    public List<String> getHeader() {
        return header;
    }

    public void setHeader(List<String> header) {
        this.header = header;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public Double getIncor_threshold() {
        return incor_threshold;
    }

    public void setIncor_threshold(Double incor_threshold) {
        this.incor_threshold = incor_threshold;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public List<Integer> getResult2() {
        return result2;
    }

    public void setResult2(List<Integer> result2) {
        this.result2 = result2;
    }

    public List<Integer> getFilter_indices() {
        return filter_indices;
    }

    public void setFilter_indices(List<Integer> filter_indices) {
        this.filter_indices = filter_indices;
    }

    public Double getFicor_threshold() {
        return ficor_threshold;
    }

    public void setFicor_threshold(Double ficor_threshold) {
        this.ficor_threshold = ficor_threshold;
    }

    public List<Double> getCorcoefficient() {
        return corcoefficient;
    }

    public void setCorcoefficient(List<Double> corcoefficient) {
        this.corcoefficient = corcoefficient;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }
}
