package model.insert;

public class CalculationDataModel {
    private String materialTrademark;

    private String mName;

    private String calSoftwareName;

    private String softwareVersion;

    private String calFormulaName;

    private String calFormulaSource;

    private String calResultFile;

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

    public String getCalSoftwareName() {
        return calSoftwareName;
    }

    public void setCalSoftwareName(String calSoftwareName) {
        this.calSoftwareName = calSoftwareName == null ? null : calSoftwareName.trim();
    }

    public String getSoftwareVersion() {
        return softwareVersion;
    }

    public void setSoftwareVersion(String softwareVersion) {
        this.softwareVersion = softwareVersion == null ? null : softwareVersion.trim();
    }

    public String getCalFormulaName() {
        return calFormulaName;
    }

    public void setCalFormulaName(String calFormulaName) {
        this.calFormulaName = calFormulaName == null ? null : calFormulaName.trim();
    }

    public String getCalFormulaSource() {
        return calFormulaSource;
    }

    public void setCalFormulaSource(String calFormulaSource) {
        this.calFormulaSource = calFormulaSource == null ? null : calFormulaSource.trim();
    }

    public String getCalResultFile() {
        return calResultFile;
    }

    public void setCalResultFile(String calResultFile) {
        this.calResultFile = calResultFile == null ? null : calResultFile.trim();
    }

    public Integer getDescriptionId() {
        return descriptionId;
    }

    public void setDescriptionId(Integer descriptionId) {
        this.descriptionId = descriptionId;
    }
}