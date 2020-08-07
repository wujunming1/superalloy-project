package entity;

import org.springframework.data.annotation.Id;

import java.util.List;

/**
 * Created by piki on 2017/11/13.
 */
public class RemainFeatureFour {
    @Id
    String id;
    String username;
    List<Double> explained_variance_ratio;
    List<Double> explained_variance;
    Integer n_components;
    String type;
    List <String> header;

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

    public List<Double> getExplained_variance_ratio() {
        return explained_variance_ratio;
    }

    public void setExplained_variance_ratio(List<Double> explained_variance_ratio) {
        this.explained_variance_ratio = explained_variance_ratio;
    }

    public List<Double> getExplained_variance() {
        return explained_variance;
    }

    public void setExplained_variance(List<Double> explained_variance) {
        this.explained_variance = explained_variance;
    }

    public Integer getN_components() {
        return n_components;
    }

    public void setN_components(Integer n_components) {
        this.n_components = n_components;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public List<String> getHeader() {
        return header;
    }

    public void setHeader(List<String> header) {
        this.header = header;
    }
}
