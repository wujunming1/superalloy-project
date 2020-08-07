package entity;

import controller.Data;

import java.util.Date;


public class LiteratureEntity {
    private String title;//文献名称
    private String author;//作者名
    private Date year;//发表时间
    private String lAbstract;//摘要，abstract关键词不能用
    private String keyWords;//关键词
    private String referenceType;//文章类型（期刊/会议/书籍等）
    private String researchInstitute;//研究机构(学校/国家)
    private int volume;//卷号
    private int issue;//期号
    private String doi;//数字对象标识
    private String materialMethod;//材料学计算方法名称
    private String mlMethod;//机器学习计算方法名称
    private String fileAttachment;//文章链接
    private int startPage;//起始页
    private int endPage;//结束页

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getAuthor() {
        return author;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public Date getYear() {
        return year;
    }

    public void setYear(Date year) {
        this.year = year;
    }

    public String getlAbstract() {
        return lAbstract;
    }

    public void setlAbstract(String lAbstract) {
        this.lAbstract = lAbstract;
    }

    public String getKeyWords() {
        return keyWords;
    }

    public void setKeyWords(String keyWords) {
        this.keyWords = keyWords;
    }

    public String getReferenceType() {
        return referenceType;
    }

    public void setReferenceType(String referenceType) {
        this.referenceType = referenceType;
    }

    public String getResearchInstitute() {
        return researchInstitute;
    }

    public void setResearchInstitute(String researchInstitute) {
        this.researchInstitute = researchInstitute;
    }

    public int getVolume() {
        return volume;
    }

    public void setVolume(int volume) {
        this.volume = volume;
    }

    public int getIssue() {
        return issue;
    }

    public void setIssue(int issue) {
        this.issue = issue;
    }

    public String getDoi() {
        return doi;
    }

    public void setDoi(String doi) {
        this.doi = doi;
    }

    public String getMaterialMethod() {
        return materialMethod;
    }

    public void setMaterialMethod(String materialMethod) {
        this.materialMethod = materialMethod;
    }

    public String getMlMethod() {
        return mlMethod;
    }

    public void setMlMethod(String mlMethod) {
        this.mlMethod = mlMethod;
    }

    public String getFileAttachment() {
        return fileAttachment;
    }

    public void setFileAttachment(String fileAttachment) {
        this.fileAttachment = fileAttachment;
    }

    public int getStartPage() {
        return startPage;
    }

    public void setStartPage(int startPage) {
        this.startPage = startPage;
    }

    public int getEndPage() {
        return endPage;
    }

    public void setEndPage(int endPage) {
        this.endPage = endPage;
    }
}
