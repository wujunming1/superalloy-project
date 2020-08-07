package entity;

import org.springframework.data.annotation.Id;

import java.util.List;

/**
 * Created by piki on 2017/11/13.
 */
public class RemainFeatureOne {
    @Id
    String id;
    String username;
    Double invar_threshold;
    List<Integer> filter_indices;
    List<Double> sparse_coef;
    Integer s_threshold;
    Double fivar_threshold;
    List<Double> variance;
    String type;
    List<Integer> result1;
    List<String> header;


    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public Double getInvar_threshold() {
        return invar_threshold;
    }

    public void setInvar_threshold(Double invar_threshold) {
        this.invar_threshold = invar_threshold;
    }

    public List<Integer> getFilter_indices() {
        return filter_indices;
    }

    public void setFilter_indices(List<Integer> filter_indices) {
        this.filter_indices = filter_indices;
    }

    public List<Double> getSparse_coef() {
        return sparse_coef;
    }

    public void setSparse_coef(List<Double> sparse_coef) {
        this.sparse_coef = sparse_coef;
    }

    public Integer getS_threshold() {
        return s_threshold;
    }

    public void setS_threshold(Integer s_threshold) {
        this.s_threshold = s_threshold;
    }

    public Double getFivar_threshold() {
        return fivar_threshold;
    }

    public void setFivar_threshold(Double fivar_threshold) {
        this.fivar_threshold = fivar_threshold;
    }

    public List<Double> getVariance() {
        return variance;
    }

    public void setVariance(List<Double> variance) {
        this.variance = variance;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public List<Integer> getResult1() {
        return result1;
    }

    public void setResult1(List<Integer> result1) {
        this.result1 = result1;
    }

    public List<String> getHeader() {
        return header;
    }

    public void setHeader(List<String> header) {
        this.header = header;
    }
}
