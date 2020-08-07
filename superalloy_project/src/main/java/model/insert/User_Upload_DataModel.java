package model.insert;

import java.util.Date;

public class User_Upload_DataModel {
    private Integer dDataDescriptionId;

    private String dAbstract;

    private String dSubmissionName;

    private String dSubmissionUnit;

    private String dProofreader;

    private Integer dSizeM;

    private String telephonenumber;

    private String email;

    private Integer dSizeN;

    private String keywords;

    private String contactAddress;

    private Date dSubmissionDate;

    private String domainType;

    private String ecode;

    private String areaType;

    private String recordName;

    private String userEcode;

    private String relatedLiterature;

    public Integer getdDataDescriptionId() {
        return dDataDescriptionId;
    }

    public void setdDataDescriptionId(Integer dDataDescriptionId) {
        this.dDataDescriptionId = dDataDescriptionId;
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

    public String getTelephonenumber() {
        return telephonenumber;
    }

    public void setTelephonenumber(String telephonenumber) {
        this.telephonenumber = telephonenumber == null ? null : telephonenumber.trim();
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email == null ? null : email.trim();
    }

    public Integer getdSizeN() {
        return dSizeN;
    }

    public void setdSizeN(Integer dSizeN) {
        this.dSizeN = dSizeN;
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

    public String getEcode() {
        return ecode;
    }

    public void setEcode(String ecode) {
        this.ecode = ecode == null ? null : ecode.trim();
    }

    public String getAreaType() {
        return areaType;
    }

    public void setAreaType(String areaType) {
        this.areaType = areaType == null ? null : areaType.trim();
    }

    public String getRecordName() {
        return recordName;
    }

    public void setRecordName(String recordName) {
        this.recordName = recordName == null ? null : recordName.trim();
    }

    public String getUserEcode() {
        return userEcode;
    }

    public void setUserEcode(String userEcode) {
        this.userEcode = userEcode == null ? null : userEcode.trim();
    }

    public String getRelatedLiterature() {
        return relatedLiterature;
    }

    public void setRelatedLiterature(String relatedLiterature) {
        this.relatedLiterature = relatedLiterature == null ? null : relatedLiterature.trim();
    }
}