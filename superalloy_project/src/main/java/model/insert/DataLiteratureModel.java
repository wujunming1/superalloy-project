package model.insert;

import java.util.Date;

public class DataLiteratureModel {
    private String id;

    private String type;

    private String title;
    private String titleE;

    private String author;
    private String authorE;

    private String publishDate;

    private String keywords;
    private String keywordsE;


    private String lAbstract;
    private String lAbstractE;

    private String referenceType;

    private String researchInstitute;

    private Integer volume;

    private Integer issue;

    private String doi;

    private String materialMethod;

    private String mlMethod;

    private String fileAttachment;//未加

    private Integer startPage;

    private Integer endPage;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id == null ? null : id.trim();
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type == null ? null : type.trim();
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title == null ? null : title.trim();
    }

    public String getAuthor() {
        return author;
    }

    public void setAuthor(String author) {
        this.author = author == null ? null : author.trim();
    }

    public String getPublishDate() {
        return publishDate;
    }

    public void setPublishDate(String publishDate) {
        this.publishDate = publishDate;
    }

    public String getKeywords() {
        return keywords;
    }

    public void setKeywords(String keywords) {
        this.keywords = keywords == null ? null : keywords.trim();
    }

    public String getlAbstract() {
        return lAbstract;
    }

    public void setlAbstract(String lAbstract) {
        this.lAbstract = lAbstract == null ? null : lAbstract.trim();
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

    public String getTitleE() {
        return titleE;
    }

    public void setTitleE(String titleE) {
        this.titleE = titleE;
    }

    public String getAuthorE() {
        return authorE;
    }

    public void setAuthorE(String authorE) {
        this.authorE = authorE;
    }

    public String getKeywordsE() {
        return keywordsE;
    }

    public void setKeywordsE(String keywordsE) {
        this.keywordsE = keywordsE;
    }

    public String getlAbstractE() {
        return lAbstractE;
    }

    public void setlAbstractE(String lAbstractE) {
        this.lAbstractE = lAbstractE;
    }
}