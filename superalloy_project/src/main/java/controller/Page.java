package controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

import static org.springframework.web.bind.annotation.RequestMethod.*;

/**
 * Created by piki on 2017/9/14.
 */
@Controller
@RequestMapping("/page")
public class Page {

    @RequestMapping(value = "/index", method = GET)
    public String index() {

        return "index";
    }

    @RequestMapping(value = "/loginPage", method = GET)
    public String login() {

        return "admin/login";
    }

    @RequestMapping(value = "/registerPage", method = GET)
    public String register() {

        return "admin/register";
    }
    @RequestMapping(value = "/task", method = GET)
    public String tasks() {

        return "admin/tasklist";
    }
    @RequestMapping(value = "/sourcedata", method = GET)
    public String datasource() {

        return "admin/datasource";
    }
    @RequestMapping(value = "/changePassword", method = GET)
    public String changePassword() {

        return "admin/changepassword";
    }
    @RequestMapping(value = "/profile", method = GET)
    public String profile() {

        return "admin/profile";
    }
    @RequestMapping(value = "/machineLearningIndex",method = GET)
    public String  machineLearningIndex(){
        return "machineLearning/machineLearning";
    }

    @RequestMapping(value = "/dataShow",method = GET)
    public String  dataShow(){
        return "machineLearning/dataShow";
    }

    @RequestMapping(value = "/featureSelect",method = GET)
    public String  featureSelect(){
        return "machineLearning/machineLearning";
    }

    @RequestMapping(value = "/attributeCalc",method = GET)
    public String  attributeCalc(){
        return "machineLearning/attributeCalc";
    }

    @RequestMapping(value = "/machineLearningItems",method = GET)
    public String  machineLearningItems(){
        return "machineLearning/machineLearningItems";
    }


    @RequestMapping(value="/machineLearning",method = GET)
    public String machineLearning(){
        return "machineLearning/data_driven_ML";
    }
    @RequestMapping(value = "/machineLearningProcess",method = GET)
    public String  machineLearningProcess(){
        return "machineLearning/machineLearningProcess";
    }

    @RequestMapping(value = "/machineLearningFullAuto",method = GET)
    public String  machineLearningFullAuto(){
        return "machineLearning/fullAutoMachineLearning";
    }

    @RequestMapping(value = "/MaterialPrediction",method = GET)
    public String  machineLearningHalfAuto(){
        return "machineLearning/MaterialPrediction";
    }
    @RequestMapping(value = "/MaterialPredictionProcess",method = GET)
    public String  materialPredictionProcess(){
        return "machineLearning/MaterialPredictionProcess";
    }
    @RequestMapping(value = "/MaterialPredictionProcessResult",method = GET)
    public String  materialPredictionProcessResult(){
        return "machineLearning/MaterialPredictionProcessResult";
    }
    @RequestMapping(value = "/MaterialPredictionProcessResult1",method = GET)
    public String  materialPredictionProcessResult1(){
        return "machineLearning/MaterialPredictionProcessResult1";
    }
    @RequestMapping(value = "/MaterialPredictionProcessResult2",method = GET)
    public String  materialPredictionProcessResult2(){
        return "machineLearning/MaterialPredictionProcessResult2";
    }
    @RequestMapping(value = "/MaterialDesignProcess",method = GET)
    public String  materialDesignProcess(){
        return "machineLearning/MaterialDesignProcess";

    }
    @RequestMapping(value = "/MaterialDesignProcessResult",method = GET)
    public String materialDesignProcessResult(){
        return "machineLearning/MaterialDesignProcessResult";}
    @RequestMapping(value = "/learningItems",method = GET)
    public String learningItems(){
        return "machineLearning/machineLearningItems";}
    @RequestMapping(value = "/MaterialDesignProcessResult1",method = GET)
    public String  materialDesignProcessResult1(){
        return "machineLearning/MaterialDesignProcessResult1";}
    @RequestMapping(value = "/MaterialDesignProcessResult2",method = GET)
    public String  materialDesignProcessResult2(){
        return "machineLearning/MaterialDesignProcessResult2";}
    @RequestMapping(value = "/dataUpload",method = GET)
    public String  dataUpload(){
        return "machineLearning/dataUpload";
    }

    @RequestMapping(value = "/contactWithCom",method = GET)
    public String  contactWithCom(){
        return "machineLearning/machineLearningList";
    }

    @RequestMapping(value = "/machineLearningList",method = GET)
    public String  machineLearningList(){
        return "machineLearning/machineLearningList_1";
    }

    @RequestMapping(value = "/oneKeyFeatureSelect",method = GET)
    public String  oneKeyFeatureSelect(){
        return "machineLearning/oneKeyFeatureSelect";
    }

    @RequestMapping(value = "/comment",method = GET)
    public String  comment(){
        return "admin/tasklist";
    }
    @RequestMapping(value = "/data_Relation",method = GET)
    public String  data_relation(){
        return "machineLearning/data_Relation";
    }

    @RequestMapping(value = "/machineLearningResult",method = GET)
    public String  machineLearningResult(){
        return "machineLearning/machineLearningResult";
    }

    @RequestMapping(value = "/machineLearningParam",method = GET)
    public String  machineLearningParam(){
        return "machineLearning/machineLearningParam";
    }

    @RequestMapping(value = "/metaFeatures", method = GET)
    public String metaFeatures(){ return "automl/cluster/selection/metaFeatures"; }

    @RequestMapping(value = "/Recommend_RF", method = GET)
    public String Recommend_RF(){ return "automl/cluster/selection/Recommend_RF"; }

    @RequestMapping(value = "/autoClusterSelectionResult", method = GET)
    public String autoClusterSelectionResult(){ return "automl/cluster/selection/autoClusterSelectionResult"; }

    @RequestMapping(value = "/Recommend_UIM", method = GET)
    public String Recommend_UIM(){ return "automl/cluster/selection/Recommend_UIM"; }

    @RequestMapping(value = "/clusterlist",method = GET)
    public String  clusterlist(){
        return "automl/cluster/integration/clusterlist";
    }
    @RequestMapping(value = "/resultshow",method = GET)
    public String  resultshow(){
        return "automl/cluster/integration/resultshow";
    }
    @RequestMapping(value = "/metaFeatures_classifier", method = GET)
    public String metaFeatures_classifier(){ return "automl/classifier/selection/metaFeatures_classifier"; }

    @RequestMapping(value = "/registerExperiments", method = GET)
    public String  registerExperiments(){
        return "machineLearning/halfAutoMachineLearningRegister";
    }
    @RequestMapping(value = "/designExperiments", method = GET)
    public String  designExperiments(){
        return "machineLearning/halfAutoMachineLearning";
    }
    @RequestMapping(value = "/executeExperiments", method = GET)
    public String  executeExperiments(){
        return "machineLearning/halfAutoMachineLearningExecution";
    }
    @RequestMapping(value = "/resultExperiments", method = GET)
    public String  resultExperiments(){
        return "machineLearning/halfAutoMachineLearningResult";
    }
    @RequestMapping(value = "/autoClassifierSelectionResult", method = GET)
    public String  autoClassifierSelectionResult(){
        return "automl/classifier/selection/autoClassifierSelectionResult";
    }

    @RequestMapping(value = "/Recommend_UIM_classifier", method = GET)
    public String  Recommend_UIM_classifier(){
        return "automl/classifier/selection/Recommend_UIM_classifier";
    }

    @RequestMapping(value = "/Recommend_RF_classifier", method = GET)
    public String  Recommend_RF_classifier(){
        return "automl/classifier/selection/Recommend_RF_classifier";
    }


}
