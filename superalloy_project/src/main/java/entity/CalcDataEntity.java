package entity;

import controller.Page;

/**
 * Created by piki on 2018/1/28.
 */
public class CalcDataEntity {
    private String materialTrademark;//牌号
    private String mName;//材料名称
    private String calSoftwareName;//计算软件名称
    private String sVersion;//软件版本号
    private String calFormulaName;//计算公式名称
    private String clFormulaSource;//计算公式出处
    private String calResultFile;//计算结果文件（文件的本地路径）

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

    public String getCalSoftwareName() {
        return calSoftwareName;
    }

    public void setCalSoftwareName(String calSoftwareName) {
        this.calSoftwareName = calSoftwareName;
    }

    public String getsVersion() {
        return sVersion;
    }

    public void setsVersion(String sVersion) {
        this.sVersion = sVersion;
    }

    public String getCalFormulaName() {
        return calFormulaName;
    }

    public void setCalFormulaName(String calFormulaName) {
        this.calFormulaName = calFormulaName;
    }

    public String getClFormulaSource() {
        return clFormulaSource;
    }

    public void setClFormulaSource(String clFormulaSource) {
        this.clFormulaSource = clFormulaSource;
    }

    public String getCalResultFile() {
        return calResultFile;
    }

    public void setCalResultFile(String calResultFile) {
        this.calResultFile = calResultFile;
    }
}
