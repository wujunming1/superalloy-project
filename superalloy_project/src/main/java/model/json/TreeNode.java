package model.json;

/**
 * Created by Zhuhaokai on 2018/1/26.
 */
public class TreeNode {
    private String title;
    private String id;
    private Integer parent;
    private String value;

    public TreeNode(String title, String id, Integer parent, String value) {
        this.title = title;
        this.id = id;
        this.parent = parent;
        this.value = value;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public Integer getParent() {
        return parent;
    }

    public void setParent(Integer parent) {
        this.parent = parent;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }
}
