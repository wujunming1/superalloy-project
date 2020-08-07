package entity;

import org.springframework.data.annotation.Id;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by piki on 2017/11/13.
 */
public class RemainFeatureTwo {
    @Id
    String id;
    Double incor_threshold;
    String username;
    List<Integer> result2;
    List<Integer> filter_indices;
    Double ficor_threshold;
    List<Double> corcoefficent;
    String type;
    List<String> header;
    ArrayList<ArrayList<Object>> relation;

    public List<Double> getCorcoefficent() {
        return corcoefficent;
    }

    public void setCorcoefficent(List<Double> corcoefficent) {
        this.corcoefficent = corcoefficent;
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

    public ArrayList<ArrayList<Object>> getRelation() {
        return relation;
    }

    public void setRelation(ArrayList<ArrayList<Object>> relation) {
        this.relation = relation;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }
}
