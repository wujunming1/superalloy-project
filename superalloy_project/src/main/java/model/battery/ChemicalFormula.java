package model.battery;

import java.util.List;

/**
 * Created by Zhuhaokai on 2017/12/3.
 */
public class ChemicalFormula {
    private String chemicalFormulaStructural;
    private String chemicalNameSystematic;
    private String chemicalFormulaSum;
    private String chemicalNameCommon;
    private String chemicalElement;
    private List<String> chemicalProportion;
    private String chemicalFixed;

    public String getChemicalFixed() {
        return chemicalFixed;
    }

    public void setChemicalFixed(String chemicalFixed) {
        this.chemicalFixed = chemicalFixed;
    }

    public String getChemicalElement() {
        return chemicalElement;
    }

    public void setChemicalElement(String chemicalElement) {
        this.chemicalElement = chemicalElement;
    }

    public String getChemicalFormulaStructural() {
        return chemicalFormulaStructural;
    }

    public void setChemicalFormulaStructural(String chemicalFormulaStructural) {
        this.chemicalFormulaStructural = chemicalFormulaStructural;
    }

    public String getChemicalNameSystematic() {
        return chemicalNameSystematic;
    }

    public void setChemicalNameSystematic(String chemicalNameSystematic) {
        this.chemicalNameSystematic = chemicalNameSystematic;
    }

    public String getChemicalFormulaSum() {
        return chemicalFormulaSum;
    }

    public void setChemicalFormulaSum(String chemicalFormulaSum) {
        this.chemicalFormulaSum = chemicalFormulaSum;
    }

    public String getChemicalNameCommon() {
        return chemicalNameCommon;
    }

    public void setChemicalNameCommon(String chemicalNameCommon) {
        this.chemicalNameCommon = chemicalNameCommon;
    }

    public List<String> getChemicalProportion() {
        return chemicalProportion;
    }

    public void setChemicalProportion(List<String> chemicalProportion) {
        this.chemicalProportion = chemicalProportion;
    }
}
