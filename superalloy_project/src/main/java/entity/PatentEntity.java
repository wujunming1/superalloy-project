package entity;

import java.util.Date;

/**
 *专利实体类
 * Created by Michael_Huang on 2018/10/20
 */

public class PatentEntity {
    private String title;//专利标题
    private String author;//作者名
    private Date year;//发表时间
    private String lAbstract;//摘要，abstract关键词不能用
    private String keyWords;//关键词
    private String patentType;//专利类型
    private String patentNumber;//专利号
    private String patentInstitute;//专利申请机构(学校/国家)

    private String patentRegion;//申请地区


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

    public String getPatentType() {
        return patentType;
    }

    public void setPatentType(String patentType) {
        this.patentType = patentType;
    }

    public String getPatentInstitute() {
        return patentInstitute;
    }

    public void setPatentInstitute(String patentInstitute) {
        this.patentInstitute = patentInstitute;
    }

    public String getPatentNumber() {
        return patentNumber;
    }

    public void setPatentNumber(String patentNumber) {
        this.patentNumber = patentNumber;
    }

    public String getPatentRegion() {
        return patentRegion;
    }

    public void setPatentRegion(String patentRegion) {
        this.patentRegion = patentRegion;
    }
}
