package model.battery;

/**
 * Created by Zhuhaokai on 2017/12/3.
 */
public class SymmetryEquivOp {
    private int symmetryEquivPosSiteId;

    public int getSymmetryEquivPosSiteId() {
        return symmetryEquivPosSiteId;
    }

    public void setSymmetryEquivPosSiteId(int symmetryEquivPosSiteId) {
        this.symmetryEquivPosSiteId = symmetryEquivPosSiteId;
    }

    public String getSymmetryEquivPosAsXyz() {
        return symmetryEquivPosAsXyz;
    }

    public void setSymmetryEquivPosAsXyz(String symmetryEquivPosAsXyz) {
        this.symmetryEquivPosAsXyz = symmetryEquivPosAsXyz;
    }

    private String symmetryEquivPosAsXyz;
}
