package model.battery;

import java.util.Date;
import java.util.zip.DataFormatException;

/**
 * Created by Zhuhaokai on 2017/12/3.
 */
public class CreateDate {
    private String auditUpdateRecord;
    private String auditCreationDate;

    public String getAuditUpdateRecord() {
        return auditUpdateRecord;
    }

    public void setAuditUpdateRecord(String auditUpdateRecord) {
        this.auditUpdateRecord = auditUpdateRecord;
    }

    public String getAuditCreationDate() {
        return auditCreationDate;
    }

    public void setAuditCreationDate(String auditCreationDate) {
        this.auditCreationDate = auditCreationDate;
    }
}
