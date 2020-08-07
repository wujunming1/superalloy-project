package model;

/**
 * Created by Zhuhaokai on 2017/12/25.
 */
public class BatteryResponse {
    private String chemicalFormulaSum;
    private String symmetrySpaceGroupNameHM;

    public BatteryResponse(String chemicalFormulaSum, String symmetrySpaceGroupNameHM) {
        this.chemicalFormulaSum = chemicalFormulaSum;
        this.symmetrySpaceGroupNameHM = symmetrySpaceGroupNameHM;
    }

    public String getChemicalFormulaSum() {
        return chemicalFormulaSum;
    }

    public void setChemicalFormulaSum(String chemicalFormulaSum) {
        this.chemicalFormulaSum = chemicalFormulaSum;
    }

    public String getSymmetrySpaceGroupNameHM() {
        return symmetrySpaceGroupNameHM;
    }

    public void setSymmetrySpaceGroupNameHM(String symmetrySpaceGroupNameHM) {
        this.symmetrySpaceGroupNameHM = symmetrySpaceGroupNameHM;
    }
}
