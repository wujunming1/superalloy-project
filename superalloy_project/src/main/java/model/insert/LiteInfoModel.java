package model.insert;

import java.util.Date;

public class LiteInfoModel {
    private Integer id;

    private String titlee;

    private String authore;

    private Date publishDate;

    private String keywordse;

    private String lAbstracte;

    private String referenceType;

    private String researchInstitute;

    private Integer volume;

    private Integer issue;

    private String doi;

    private String materialMethod;

    private String mlMethod;

    private String fileAttachment;

    private Integer startPage;

    private Integer endPage;

    private Integer relatedDataId;

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getTitlee() {
        return titlee;
    }

    public void setTitlee(String titlee) {
        this.titlee = titlee == null ? null : titlee.trim();
    }

    public String getAuthore() {
        return authore;
    }

    public void setAuthore(String authore) {
        this.authore = authore == null ? null : authore.trim();
    }

    public Date getPublishDate() {
        return publishDate;
    }

    public void setPublishDate(Date publishDate) {
        this.publishDate = publishDate;
    }

    public String getKeywordse() {
        return keywordse;
    }

    public void setKeywordse(String keywordse) {
        this.keywordse = keywordse == null ? null : keywordse.trim();
    }

    public String getlAbstracte() {
        return lAbstracte;
    }

    public void setlAbstracte(String lAbstracte) {
        this.lAbstracte = lAbstracte == null ? null : lAbstracte.trim();
    }

    public String getReferenceType() {
        return referenceType;
    }

    public void setReferenceType(String referenceType) {
        this.referenceType = referenceType == null ? null : referenceType.trim();
    }

    public String getResearchInstitute() {
        return researchInstitute;
    }

    public void setResearchInstitute(String researchInstitute) {
        this.researchInstitute = researchInstitute == null ? null : researchInstitute.trim();
    }

    public Integer getVolume() {
        return volume;
    }

    public void setVolume(Integer volume) {
        this.volume = volume;
    }

    public Integer getIssue() {
        return issue;
    }

    public void setIssue(Integer issue) {
        this.issue = issue;
    }

    public String getDoi() {
        return doi;
    }

    public void setDoi(String doi) {
        this.doi = doi == null ? null : doi.trim();
    }

    public String getMaterialMethod() {
        return materialMethod;
    }

    public void setMaterialMethod(String materialMethod) {
        this.materialMethod = materialMethod == null ? null : materialMethod.trim();
    }

    public String getMlMethod() {
        return mlMethod;
    }

    public void setMlMethod(String mlMethod) {
        this.mlMethod = mlMethod == null ? null : mlMethod.trim();
    }

    public String getFileAttachment() {
        return fileAttachment;
    }

    public void setFileAttachment(String fileAttachment) {
        this.fileAttachment = fileAttachment == null ? null : fileAttachment.trim();
    }

    public Integer getStartPage() {
        return startPage;
    }

    public void setStartPage(Integer startPage) {
        this.startPage = startPage;
    }

    public Integer getEndPage() {
        return endPage;
    }

    public void setEndPage(Integer endPage) {
        this.endPage = endPage;
    }

    public Integer getRelatedDataId() {
        return relatedDataId;
    }

    public void setRelatedDataId(Integer relatedDataId) {
        this.relatedDataId = relatedDataId;
    }
}