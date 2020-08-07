package entity;

import controller.Data;

import java.lang.ref.PhantomReference;
import java.util.Date;

/**
 * 数据的描述（元数据）
 * Created by piki on 2018/1/28.
 */
public class DataDescriptionEntity {
    private String dAbstract;//数据集摘要
    private String dSubmissionName;//数据提交者
    private String dSubmissionUnit;//数据提交者所在单位
    private String dProofReader;//数据校对者
    private int dSizeM;//数据集大小（格式为m个样本，维度为n）
    private int dSizeN;
    private String email;//邮箱
    private String telephoneNumber;//手机号或固话
    private String keyWords;//关键词
    private String contactAddress;//通讯地址
    private Date dSubmissionDate;//数据提交日期（日期格式YYYY-MM-dd）
    private String dSourceInfo;//数据来源信息
    private String domainType;//数据邻域类型
    private String areaType;//研究方向类型

    public String getdAbstract() {
        return dAbstract;
    }

    public void setdAbstract(String dAbstract) {
        this.dAbstract = dAbstract;
    }

    public String getdSubmissionName() {
        return dSubmissionName;
    }

    public void setdSubmissionName(String dSubmissionName) {
        this.dSubmissionName = dSubmissionName;
    }

    public String getdSubmissionUnit() {
        return dSubmissionUnit;
    }

    public void setdSubmissionUnit(String dSubmissionUnit) {
        this.dSubmissionUnit = dSubmissionUnit;
    }

    public String getdProofReader() {
        return dProofReader;
    }

    public void setdProofReader(String dProofReader) {
        this.dProofReader = dProofReader;
    }

    public int getdSizeM() {
        return dSizeM;
    }

    public void setdSizeM(int dSizeM) {
        this.dSizeM = dSizeM;
    }

    public int getdSizeN() {
        return dSizeN;
    }

    public void setdSizeN(int dSizeN) {
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

    public String getKeyWords() {
        return keyWords;
    }

    public void setKeyWords(String keyWords) {
        this.keyWords = keyWords;
    }

    public String getContactAddress() {
        return contactAddress;
    }

    public void setContactAddress(String contactAddress) {
        this.contactAddress = contactAddress;
    }

    public Date getdSubmissionDate() {
        return dSubmissionDate;
    }

    public void setdSubmissionDate(Date dSubmissionDate) {
        this.dSubmissionDate = dSubmissionDate;
    }

    public String getdSourceInfo() {
        return dSourceInfo;
    }

    public void setdSourceInfo(String dSourceInfo) {
        this.dSourceInfo = dSourceInfo;
    }

    public String getDomainType() {
        return domainType;
    }

    public void setDomainType(String domainType) {
        this.domainType = domainType;
    }

    public String getAreaType() {
        return areaType;
    }

    public void setAreaType(String areaType) {
        this.areaType = areaType;
    }
}
