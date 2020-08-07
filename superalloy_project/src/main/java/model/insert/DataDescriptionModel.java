package model.insert;

import java.util.Date;

public class DataDescriptionModel {
    private String dAbstract;

    private String dSubmissionName;

    private String dSubmissionUnit;

    private String dProofreader;

    private Integer dSizeM;

    private Integer dSizeN;

    private String email;

    private String telephoneNumber;


    private String keywords;

    private String contactAddress;

    private Date dSubmissionDate;
    
    //private String Dsourceinfo;

    private String domainType;


    private String areaType;

    private Integer dDataDescriptionId;
    
    private String eCode;

    private String recordName;

    private String userECode;
    private String relatedLiterature;
    public String getRelatedLiterature() {
        return relatedLiterature;
    }
    public void setRelatedLiterature(String relatedLiterature) {
        this.relatedLiterature = relatedLiterature;
    }
    public String getdAbstract() {
        return dAbstract;
    }

    public void setdAbstract(String dAbstract) {
        this.dAbstract = dAbstract == null ? null : dAbstract.trim();
    }

    public String getdSubmissionName() {
        return dSubmissionName;
    }

    public void setdSubmissionName(String dSubmissionName) {
        this.dSubmissionName = dSubmissionName == null ? null : dSubmissionName.trim();
    }

    public String getdSubmissionUnit() {
        return dSubmissionUnit;
    }

    public void setdSubmissionUnit(String dSubmissionUnit) {
        this.dSubmissionUnit = dSubmissionUnit == null ? null : dSubmissionUnit.trim();
    }

    public String getdProofreader() {
        return dProofreader;
    }

    public void setdProofreader(String dProofreader) {
        this.dProofreader = dProofreader == null ? null : dProofreader.trim();
    }

    public Integer getdSizeM() {
        return dSizeM;
    }

    public void setdSizeM(Integer dSizeM) {
        this.dSizeM = dSizeM;
    }

    public Integer getdSizeN() {
        return dSizeN;
    }

    public void setdSizeN(Integer dSizeN) {
        this.dSizeN = dSizeN;
    }


    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getTelephoneNumber() {
        return telephoneNumber;
    }

    public void setTelephoneNumber(String telephoneNumber) {
        this.telephoneNumber = telephoneNumber;
    }

    public String getKeywords() {
        return keywords;
    }

    public void setKeywords(String keywords) {
        this.keywords = keywords == null ? null : keywords.trim();
    }

    public String getContactAddress() {
        return contactAddress;
    }

    public void setContactAddress(String contactAddress) {
        this.contactAddress = contactAddress == null ? null : contactAddress.trim();
    }

    public Date getdSubmissionDate() {
        return dSubmissionDate;
    }

    public void setdSubmissionDate(Date dSubmissionDate) {
        this.dSubmissionDate = dSubmissionDate;
    }

    public String getDomainType() {
        return domainType;
    }

    public void setDomainType(String domainType) {
        this.domainType = domainType == null ? null : domainType.trim();
    }



    public String getAreaType() {
        return areaType;
    }

    public void setAreaType(String areaType) {
        this.areaType = areaType == null ? null : areaType.trim();
    }

    public Integer getdDataDescriptionId() {
        return dDataDescriptionId;
    }

    public void setdDataDescriptionId(Integer dDataDescriptionId) {
        this.dDataDescriptionId = dDataDescriptionId;
    }

    public String geteCode() {
        return eCode;
    }

    public void seteCode(String eCode) {
        this.eCode = eCode;
    }

    public String getRecordName() {
        return recordName;
    }

    public void setRecordName(String recordName) {
        this.recordName = recordName;
    }

    public String getUserECode() {
        return userECode;
    }

    public void setUserECode(String userECode) {
        this.userECode = userECode;
    }
}