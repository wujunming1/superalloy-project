package model.insert;

public class ExperimentDataModel {
    private String materialTrademark;

    private String mName;

    private String expconName;

    private String expParasetting;

    private String expDeviceName;

    private String remark;

    private Integer descriptionId;

    public String getMaterialTrademark() {
        return materialTrademark;
    }

    public void setMaterialTrademark(String materialTrademark) {
        this.materialTrademark = materialTrademark == null ? null : materialTrademark.trim();
    }

    public String getmName() {
        return mName;
    }

    public void setmName(String mName) {
        this.mName = mName == null ? null : mName.trim();
    }

    public String getExpconName() {
        return expconName;
    }

    public void setExpconName(String expconName) {
        this.expconName = expconName == null ? null : expconName.trim();
    }

    public String getExpParasetting() {
        return expParasetting;
    }

    public void setExpParasetting(String expParasetting) {
        this.expParasetting = expParasetting == null ? null : expParasetting.trim();
    }

    public String getExpDeviceName() {
        return expDeviceName;
    }

    public void setExpDeviceName(String expDeviceName) {
        this.expDeviceName = expDeviceName == null ? null : expDeviceName.trim();
    }

    public String getRemark() {
        return remark;
    }

    public void setRemark(String remark) {
        this.remark = remark == null ? null : remark.trim();
    }

    public Integer getDescriptionId() {
        return descriptionId;
    }

    public void setDescriptionId(Integer descriptionId) {
        this.descriptionId = descriptionId;
    }
}