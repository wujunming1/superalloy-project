package model.battery;

/**
 * Created by Zhuhaokai on 2017/12/3.
 */
public class SpaceGroup {
    private String symmetrySpaceGroupNameHall;
    private int spaceGroupItNumber;
    private String symmetryCellSetting;
    private String symmetrySpaceGroupNameHM;
    private int symmetryIntTablesNumber;

    public String getSymmetrySpaceGroupNameHall() {
        return symmetrySpaceGroupNameHall;
    }

    public void setSymmetrySpaceGroupNameHall(String symmetrySpaceGroupNameHall) {
        this.symmetrySpaceGroupNameHall = symmetrySpaceGroupNameHall;
    }

    public int getSpaceGroupItNumber() {
        return spaceGroupItNumber;
    }

    public void setSpaceGroupItNumber(int spaceGroupItNumber) {
        this.spaceGroupItNumber = spaceGroupItNumber;
    }

    public String getSymmetryCellSetting() {
        return symmetryCellSetting;
    }

    public void setSymmetryCellSetting(String symmetryCellSetting) {
        this.symmetryCellSetting = symmetryCellSetting;
    }

    public String getSymmetrySpaceGroupNameHM() {
        return symmetrySpaceGroupNameHM;
    }

    public void setSymmetrySpaceGroupNameHM(String symmetrySpaceGroupNameHM) {
        this.symmetrySpaceGroupNameHM = symmetrySpaceGroupNameHM;
    }

    public int getSymmetryIntTablesNumber() {
        return symmetryIntTablesNumber;
    }

    public void setSymmetryIntTablesNumber(int symmetryIntTablesNumber) {
        this.symmetryIntTablesNumber = symmetryIntTablesNumber;
    }
}

