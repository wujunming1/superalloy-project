package entity;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

/**
 * Created by piki on 2017/10/29.
 */
@Document
public class SysAccount {
    @Id
    private String id;
    private String username; //用户名
    private String password;//密码
    private String occupation;
    private String resDomain;
    private String resDirection;
    private String contact;
    private String email;
    private String address;
    private String unit;
    private String userECode;

    public String getUserECode() {
        return userECode;
    }

    public void setUserECode(String userECode) {
        this.userECode = userECode;
    }



    public SysAccount(String username,String password){
        this.username=username;
        this.password=password;
    }
    public SysAccount(){

    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getContact() {
        return contact;
    }

    public void setContact(String contact) {
        this.contact = contact;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getUnit() {
        return unit;
    }

    public void setUnit(String unit) {
        this.unit = unit;
    }



    public String getOccupation() {
        return occupation;
    }

    public void setOccupation(String occupation) {
        this.occupation = occupation;
    }

    public String getResDomain() {
        return resDomain;
    }

    public void setResDomain(String resDomain) {
        this.resDomain = resDomain;
    }

    public String getResDirection() {
        return resDirection;
    }

    public void setResDirection(String resDirection) {
        this.resDirection = resDirection;
    }

}
