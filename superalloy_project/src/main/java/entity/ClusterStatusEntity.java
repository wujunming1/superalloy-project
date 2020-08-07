package entity;
import org.springframework.data.annotation.Id;
import java.util.List;
public class ClusterStatusEntity {
    @Id
    String id;
    String date;
    String type;
    String username;
    String data_name;
    Integer status;

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

    public void setStatus(Integer status) {
        this.status = status;
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

    public Integer getStatus() {
        return status;
    }
}
