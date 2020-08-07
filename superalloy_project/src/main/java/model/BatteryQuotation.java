package model;

/**
 * Created by Zhuhaokai on 2017/12/24.
 */
public class BatteryQuotation {
    private String index;
    private String type;

    private String elements;
    private String spaceGroup;

    private String[] mustElements;
    private String[] notElements;
    private String shouldElement;

    private int size;
    private int page;

    public String getIndex() {
        return index;
    }

    public void setIndex(String index) {
        this.index = index;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getElements() {
        return elements;
    }

    public void setElements(String elements) {
        this.elements = elements;
    }

    public String[] getMustElements() {
        return mustElements;
    }

    public void setMustElements(String[] mustElements) {
        this.mustElements = mustElements;
    }

    public String[] getNotElements() {
        return notElements;
    }

    public void setNotElements(String[] notElements) {
        this.notElements = notElements;
    }

    public String getShouldElement() {
        return shouldElement;
    }

    public void setShouldElement(String shouldElement) {
        this.shouldElement = shouldElement;
    }

    public String getSpaceGroup() {
        return spaceGroup;
    }

    public void setSpaceGroup(String spaceGroup) {
        this.spaceGroup = spaceGroup;
    }

    public int getSize() {
        return size;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public int getPage() {
        return page;
    }

    public void setPage(int page) {
        this.page = page;
    }

    public boolean isElementsEmpty(){
        return (notElements.length == 0) && (shouldElement.equals("")) && (mustElements.length == 0);
    }
}
