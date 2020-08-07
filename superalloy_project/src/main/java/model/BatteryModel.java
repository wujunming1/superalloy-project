package model;

import model.battery.*;

import java.util.Date;
import java.util.List;

/**
 * Created by Zhuhaokai on 2017/12/3.
 */
public class BatteryModel {
    private String id;
    private ChemicalFormula chemicalFormula;
    private List<AtomsOxidation> atomsOxidations;
    private List<Site> sites;
    private List<SymmetryEquivOp> symmetryEquivOps;
    private List<Reference> references;
    private CreateDate createDate;
    private SpaceGroup spaceGroup;
    private Lattice lattice;
    private String publAuthorName;
    private String origin;
    private String content;
    private int originId;
    private String publSectionTitle;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public ChemicalFormula getChemicalFormula() {
        return chemicalFormula;
    }

    public void setChemicalFormula(ChemicalFormula chemicalFormula) {
        this.chemicalFormula = chemicalFormula;
    }

    public List<AtomsOxidation> getAtomsOxidations() {
        return atomsOxidations;
    }

    public void setAtomsOxidations(List<AtomsOxidation> atomsOxidations) {
        this.atomsOxidations = atomsOxidations;
    }

    public List<Site> getSites() {
        return sites;
    }

    public void setSites(List<Site> sites) {
        this.sites = sites;
    }

    public List<SymmetryEquivOp> getSymmetryEquivOps() {
        return symmetryEquivOps;
    }

    public void setSymmetryEquivOps(List<SymmetryEquivOp> symmetryEquivOps) {
        this.symmetryEquivOps = symmetryEquivOps;
    }

    public List<Reference> getReferences() {
        return references;
    }

    public void setReferences(List<Reference> references) {
        this.references = references;
    }

    public CreateDate getCreateDate() {
        return createDate;
    }

    public void setCreateDate(CreateDate createDate) {
        this.createDate = createDate;
    }

    public SpaceGroup getSpaceGroup() {
        return spaceGroup;
    }

    public void setSpaceGroup(SpaceGroup spaceGroup) {
        this.spaceGroup = spaceGroup;
    }

    public Lattice getLattice() {
        return lattice;
    }

    public void setLattice(Lattice lattice) {
        this.lattice = lattice;
    }

    public String getPublAuthorName() {
        return publAuthorName;
    }

    public void setPublAuthorName(String publAuthorName) {
        this.publAuthorName = publAuthorName;
    }

    public String getOrigin() {
        return origin;
    }

    public void setOrigin(String origin) {
        this.origin = origin;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public int getOriginId() {
        return originId;
    }

    public void setOriginId(int originId) {
        this.originId = originId;
    }

    public String getPublSectionTitle() {
        return publSectionTitle;
    }

    public void setPublSectionTitle(String publSectionTitle) {
        this.publSectionTitle = publSectionTitle;
    }

}
