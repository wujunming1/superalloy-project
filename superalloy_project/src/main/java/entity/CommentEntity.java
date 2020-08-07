package entity;

public class CommentEntity {
    int reliable;
    int complete;
    int helpful;
    String comment;
    String eCode;

    public int getReliable() {
        return reliable;
    }

    public void setReliable(int reliable) {
        this.reliable = reliable;
    }

    public int getComplete() {
        return complete;
    }

    public void setComplete(int complete) {
        this.complete = complete;
    }

    public int getHelpful() {
        return helpful;
    }

    public void setHelpful(int helpful) {
        this.helpful = helpful;
    }

    public String getComment() {
        return comment;
    }

    public void setComment(String comment) {
        this.comment = comment;
    }

    public String geteCode() {
        return eCode;
    }

    public void seteCode(String eCode) {
        this.eCode = eCode;
    }
}
