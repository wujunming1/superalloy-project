package model.insert;

public class DataPatentModel {
    private String id;

    private String type;

    private String title;
    private String titleE;

    private String author;
    private String authorE;

    private String publishDate;

    private String keywords;
    private String keywordsE;

    private String fileAttachment;//未加

    private String lAbstract;
    private String lAbstractE;

    private String patentType;

    private String patentInstitute;

    private String patentNumber;

    private String patentRegion;

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

    public String getPatentType() {
        return patentType;
    }

    public void setPatentType(String patentType) {
        this.patentType = patentType == null ? null : patentType.trim();
    }

    public String getPatentInstitute() {
        return patentInstitute;
    }

    public void setPatentInstitute(String patentInstitute) {
        this.patentInstitute = patentInstitute == null ? null : patentInstitute.trim();
    }


    public String getPatentNumber() {
        return patentNumber;
    }

    public void setPatentNumber(String patentNumber) {
        this.patentNumber = patentNumber == null ? null : patentNumber.trim();
    }

    public String getPatentRegion() {
        return patentRegion;
    }

    public void setPatentRegion(String patentRegion) {
        this.patentRegion = patentRegion == null ? null : patentRegion.trim();
    }

    public String getFileAttachment() {
        return fileAttachment;
    }

    public void setFileAttachment(String fileAttachment) {
        this.fileAttachment = fileAttachment == null ? null : fileAttachment.trim();
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
