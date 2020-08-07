package entity;

import org.springframework.data.annotation.Id;

import java.util.List;

/**
 * Created by piki on 2017/11/13.
 */
public class RemainFeatureThree {
    @Id
    String id;
    String username;
    List<Double> importance;
    List<Integer> result3;
    List<Integer> filter_indices;
    String  type;
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

    public List<Double> getImportance() {
        return importance;
    }

    public void setImportance(List<Double> importance) {
        this.importance = importance;
    }

    public List<Integer> getResult3() {
        return result3;
    }

    public void setResult3(List<Integer> result3) {
        this.result3 = result3;
    }

    public List<Integer> getFilter_indices() {
        return filter_indices;
    }

    public void setFilter_indices(List<Integer> filter_indices) {
        this.filter_indices = filter_indices;
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
