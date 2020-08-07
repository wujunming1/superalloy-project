package entity;

/**
 * Created by piki on 2018/1/28.
 */
public class ExpDataEntity {
    private String materialTrademark;//牌号
    private String mName;//材料名称
    private String expconName;//实验条件名称
    private String expParasetting;//实验参数设置（温度、压强、时间、速度等）
    private String expDeviceName;//实验设备名称与型号
    private String remark;//备注

    public String getMaterialTrademark() {
        return materialTrademark;
    }

    public void setMaterialTrademark(String materialTrademark) {
        this.materialTrademark = materialTrademark;
    }

    public String getmName() {
        return mName;
    }

    public void setmName(String mName) {
        this.mName = mName;
    }

    public String getExpconName() {
        return expconName;
    }

    public void setExpconName(String expconName) {
        this.expconName = expconName;
    }

    public String getExpParasetting() {
        return expParasetting;
    }

    public void setExpParasetting(String expParasetting) {
        this.expParasetting = expParasetting;
    }

    public String getExpDeviceName() {
        return expDeviceName;
    }

    public void setExpDeviceName(String expDeviceName) {
        this.expDeviceName = expDeviceName;
    }

    public String getRemark() {
        return remark;
    }

    public void setRemark(String remark) {
        this.remark = remark;
    }
}
