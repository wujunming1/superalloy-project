<%--
  Created by IntelliJ IDEA.
  User: buming
  Date: 2019/4/12
  Time: 06:03
  To change this template use File | Settings | File Templates.
--%>
<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8" %>

<%@include file="/WEB-INF/common/tag.jsp" %>

<!DOCTYPE html>
<html lang="en－zh" class="no-js">
<!--<html lang="en－zh" class="no-js">-->
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="author" content="shu">
    <meta name="Description" content="高温合金机器学习平台"/>
    <%--<script type="text/javascript" src="${ctx}/static/static/layui/layui.js"></script>--%>
    <%@include file="/WEB-INF/common/head.jsp" %>
    <title>高温合金机器学习平台</title>
    <%@include file="/WEB-INF/common/footScript.jsp" %>

    <style>
        nav{
            height: 33px
        }
        .my-nav>ul>li.current>a{
            font-weight:600;width: 139px;background-image:  url(${ctx}/pic/navIn.png);background-size:139px 34px ;background-repeat: no-repeat
        }
        .my-nav>ul>li>a{
            font-weight:600;width: 139px;background-image:  url(${ctx}/pic/navUnin.png);background-size:139px 34px ;background-repeat: no-repeat
        }
    </style>
</head>
<!-- hijacking: on/off - animation: none/scaleDown/rotate/gallery/catch/opacity/fixed/parallax -->

<body data-hijacking="off" data-animation="rotate"  style="background-color: rgb(242,242,242)">


<header class="navbar-wrapper navbar-fixed-top">
    <div class="navbar navbar-black "  style="background-image:  url(${ctx}/pic/head.jpg);height: 120px">
        <div class="container cl" style="height:120px;">
            <div class="col-sm-5" style="text-align: center;">
                <img src="${ctx}/pic/logo_03.png" style="width: 100%; height: auto;max-width: 100%; display: block;padding-top: 10px">
            </div>
            <div class="col-sm-7">
                <div style="height: 87px"></div>
                <nav class="nav navbar-nav nav-collapse my-nav" role="navigation" id="Hui-navbar">
                    <ul class="cl" style="margin: 0px;padding: 0px">
                        <li><a href="/"  style=";color: rgb(117,117,117)">首页</a></li>
                        <li><a href="${ctx}/page/dataUpload"  style=";color: rgb(117,117,117)">数据上传</a></li>
                        <li><a href="#"  style=";color: rgb(117,117,117)">联系我们</a></li>
                    </ul>
                </nav>
                <nav class="nav navbar-nav nav-collapse f-r" role="navigation" >
                    <shiro:user>
                        <ul class="cl"  style="margin: 0px;padding: 0px;bottom:0px">
                            <li><span class="logo navbar-slogan f-l mr-10 hidden-xs">账号 : <shiro:principal type="java.lang.String"/></span></li>
                            <li><a href="${ctx}/admin/logout">注销</a></li>
                        </ul>

                    </shiro:user>
                    <shiro:guest>
                        <ul class="cl"  style="margin: 0px;padding: 0px">
                            <li><a href="${ctx}/page/loginPage">登录</a></li>
                            <li><a href="${ctx}/page/registerPage">注册</a></li>
                        </ul>
                    </shiro:guest>


                </nav>
            </div>
        </div>
    </div>
</header>

<shiro:hasRole name="common">
    <section class="cd-section" style="padding-top: 50px">
        <div style="padding-top: 50px">
            <div class="container" style="margin-top: 20px;">

                <div class="row clearfix">
                    <div class="col-md-5 column form-inline">
                        <h2>AutoML-自动分类算法选择</h2>
                    </div>
                </div>
                <div class="row clearfix">
                    <div class="col-md-5 column form-inline">
                        <hr style=" height:2px;border:none;border-top:2px solid #6fb3e0;" />
                    </div>
                </div>
                <div class="row">
                    <div class="col-xs-6">
                        <div id="Echarts" style="width:1100px;height:200px;display:inline-block"></div>
                    </div>
                </div>
                <hr style=" height:0.5px;border:none;border-top:0.5px solid #6fb3e0;"/>
                <div class="panel panel-success">
                        <%--<div class="panel-heading">--%>
                        <%--<h3 class="panel-title">--%>
                        <%--推荐算法系统--%>
                        <%--</h3>--%>
                        <%--</div>--%>

                    <div class="panel-group" id="accordion">
                        <div>
                            <div class="panel-heading" >
                                <h3 class="panel-title">
                                    <div id = 'Classifier1'></div>
                                </h3>
                            </div>
                            <ul id="alg1Tab" class="nav nav-tabs">
                                <li class="active">
                                    <a href="#alg1Data" data-toggle="tab">
                                        分类结果
                                    </a>
                                </li>
                                <li>
                                    <a href="#alg1Vision" data-toggle="tab">
                                        可视化展示
                                    </a>
                                </li>
                            </ul>

                            <div id="TabContent1" class="tab-content center"></div>

                            <div id="alg1TabContent" class="tab-content">
                                <div class="tab-pane fade in active" id="alg1Data">
                                    <div class="row">
                                        <div class="col-md-12" class="table-responsive" style="overflow:auto;">
                                            <table id="alg1DataTable" class="table-responsive">

                                            </table>
                                        </div>
                                    </div>
                                </div>

                                <div class="tab-pane fade" id="alg1Vision">
                                    <div class="row">
                                        <div class="col-xs-6">
                                            <div id="alg1Echart" style="width:1000px;height:500px;display:inline-block"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="panel-heading" >
                                <h3 class="panel-title">
                                    <div id = 'Classifier2'></div>
                                </h3>
                            </div>
                            <ul id="alg2Tab" class="nav nav-tabs">
                                <li class="active">
                                    <a href="#alg2Data" data-toggle="tab">
                                        分类结果
                                    </a>
                                </li>
                                <li>
                                    <a href="#alg2Vision" data-toggle="tab">
                                        可视化展示
                                    </a>
                                </li>
                            </ul>

                            <div id="TabContent2" class="tab-content center"></div>

                            <div id="alg2TabContent" class="tab-content">
                                <div class="tab-pane fade in active" id="alg2Data">
                                    <div class="row">
                                        <div class="col-md-12" class="table-responsive" style="overflow:auto;">
                                            <table id="alg2DataTable" class="table-responsive">

                                            </table>
                                        </div>
                                    </div>
                                </div>

                                <div class="tab-pane fade" id="alg2Vision">
                                    <div class="row">
                                        <div class="col-xs-6">
                                            <div id="alg2Echart" style="width:1000px;height:500px;display:inline-block"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div class="panel-heading" >
                                <h3 class="panel-title">
                                    <div id = 'Classifier3'></div>
                                </h3>
                            </div>
                            <ul id="alg3Tab" class="nav nav-tabs">
                                <li class="active">
                                    <a href="#alg3Data" data-toggle="tab">
                                        分类结果
                                    </a>
                                </li>
                                <li>
                                    <a href="#alg3Vision" data-toggle="tab">
                                        可视化展示
                                    </a>
                                </li>
                            </ul>

                            <div id="TabContent3" class="tab-content center"></div>

                            <div id="alg3TabContent" class="tab-content">
                                <div class="tab-pane fade in active" id="alg3Data">
                                    <div class="row">
                                        <div class="col-md-12" class="table-responsive" style="overflow:auto;">
                                            <table id="alg3DataTable" class="table-responsive">

                                            </table>
                                        </div>
                                    </div>
                                </div>

                                <div class="tab-pane fade" id="alg3Vision">
                                    <div class="row">
                                        <div class="col-xs-6">
                                            <div id="alg3Echart" style="width:1000px;height:500px;display:inline-block"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                        </div>
                    </div>
                </div>


            </div>
        </div>
    </section>
</shiro:hasRole>
<script src="${ctx}/static/static/js/velocity.min.js"></script>
<script src="${ctx}/static/static/js/velocity.ui.min.js"></script>
<script src="${ctx}/static/static/js/main.js"></script> <!-- Resource jQuery -->
<script src="${ctx}/static/static/js/FileSaver.min.js"></script>
<script src="${ctx}/static/static/js/xlsx.core.min.js"></script>
<script src="${ctx}/static/static/js/tableExport.js"></script>
<script src="${ctx}/static/static/js/bootstrap-table-export.js"></script>


</body>

<script>
    var eCode= '<%=request.getParameter("eCode")%>';
    var alg1 = '<%=request.getParameter("alg1")%>';
    var alg2 = '<%=request.getParameter("alg2")%>';
    var alg3 = '<%=request.getParameter("alg3")%>';

    document.getElementById("Classifier1").innerHTML= alg1 +"分类算法";
    document.getElementById("Classifier2").innerHTML= alg2 +"分类算法";
    document.getElementById("Classifier3").innerHTML= alg3 +"分类算法";


    $(function () {
        getAlgFirst(eCode,alg1);
        getAlgSecond(eCode,alg2);
        getAlgThree(eCode,alg3);
    });

    $(function () {
        option = {
            title: {
                text: '数据驱动的自动分类算法选择模型'
            },
            tooltip: {},
            animationDurationUpdate: 1500,
            animationEasingUpdate: 'quinticInOut',
            series : [
                {
                    type: 'graph',
                    layout: 'none',
                    symbolSize: 50,
                    roam: true,
                    label: {
                        normal: {
                            show: true
                        }
                    },
                    edgeSymbol: ['circle', 'arrow'],
                    edgeSymbolSize: [4, 10],
                    edgeLabel: {
                        normal: {
                            textStyle: {
                                fontSize: 20
                            }
                        }
                    },
                    data: [{
                        name: '数据集元特征计算',
                        x: 200,
                        y: 300,
                        symbol:'circle',
                        itemStyle:{
                            color:'#969696'
                        },
                        symbolSize:125
                        // symbolSize: [100,40]
                    }, {
                        name: '初步分类算法推荐\n（三种协同过滤推荐算法）',
                        x: 550,
                        y: 300,
                        symbol:'circle',
                        itemStyle:{
                            color:'#969696'
                        },
                        symbolSize:125
                        // symbolSize: [175,40]

                    }, {
                        name: '最优分类算法推荐\n（投票集成算法）',
                        x: 1000,
                        y: 300,
                        symbol:'circle',
                        itemStyle:{
                            color:'#969696'
                        },
                        symbolSize:125
                        // symbolSize: [200,40]
                    }, {
                        name: '分类结果展示',
                        x: 1400,
                        y: 300,
                        symbol:'circle',
                        itemStyle:{
                            color:'#3399FF'
                        },
                        symbolSize:125
                        // symbolSize: [100,40]
                    }],
                    links: [{
                        source: '数据集元特征计算',
                        target: '初步分类算法推荐\n（三种协同过滤推荐算法）'
                    }, {
                        source: '初步分类算法推荐\n（三种协同过滤推荐算法）',
                        target: '最优分类算法推荐\n（投票集成算法）'
                    }, {
                        source: '最优分类算法推荐\n（投票集成算法）',
                        target: '分类结果展示'
                    }],
                    lineStyle: {
                        normal: {
                            opacity: 0.9,
                            width: 2,
                            curveness: 0
                        }
                    }
                }
            ]
        };
        var Echart = echarts.init(document.getElementById('Echarts'));
        Echart.setOption(option);
    });

    function getAlgFirst(eCode,alg1) {
        $.ajax({
            type:"get",
            url:"${pageContext.request.contextPath}/AutoMlClassifierSelection/ClassifierModel",
            data:{eCode:eCode,alg:alg1},
            dataType:"json",
            beforeSend:function(){
                $("#TabContent1").html("正在处理，请稍后···");
            },
            success:function (result) {
                $("#TabContent1").hide();

                var feature_names=result['feature_names'];
                var data=result['data'];
                var result_label=result['result_label'];
                var xy=result['xy'];


                getAlg1DataTable(feature_names,data,result_label);
                alg1Echart(result_label,xy,data,feature_names)
            },
            complete: function () {
                reloadpage();
            }
        })
    }

    function getAlg1DataTable(feature_names,data,result_label) {
        var tableTitle=[];
        var tabledata=[];
        var title = [];
        var i,j;
        for(i=0;i<feature_names.length;i++){
            title[i] = feature_names[i];
            tableTitle[i]={
                class: 'w100',
                field:''+feature_names[i],
                title:''+feature_names[i]
            };
        }
        title[i] = 'result';
        tableTitle[i] = {
            class: 'w100',
            field:'result',
            title:'result'
        };

        for(i=0;i<data.length;i++){
            var temp={};
            for(j=0;j<data[i].length;j++){
                temp[title[j]]= data[i][j];
            }
            temp[title[j]]= result_label[i];
            tabledata[i]=temp;
        }
        // console.log(tableTitle);
        // console.log(tabledata);
        // System.out.println(tableTitle);
        // System.out.println(tabledata);

        $('#alg1DataTable').bootstrapTable('destroy');
        $('#alg1DataTable').bootstrapTable({
            icons: {export: 'glyphicon-export'},
            showExport: true,
            // showRefresh: true,
            exportTypes:['csv', 'txt', 'sql', 'doc', 'excel', 'xlsx', 'pdf'],
            exportDataType: "all",
            exportOptions:{
                ignoreColumn: [0,1],  //忽略某一列的索引
                // filename: "分类结果",
                excelstyles: ['background-color', 'color', 'font-size', 'font-weight']
                // onMsoNumberFormat: DoOnMsoNumberFormat
            },
            pagination: true,
            pageSize:"15",
            columns:tableTitle,
            data: tabledata
        });
    }

    function alg1Echart(result_label,xy,data,title){
        var alldata = data;
        var alldata1=[]
        var allmax=[];
        var allmin=[];
        var allmid=[];
        var allxy = xy;
        var alltitle = title;
        var temp=[],i,j,k,n,temp1=[];
        var set_temp=new Set(result_label);
        var array_temp=Array.from(set_temp);

        for(i=0;i<array_temp.length;i++){
            var array=[];
            var list=[];
            for(j=0;j<result_label.length;j++){
                if(result_label[j]==array_temp[i]){
                    array.push(xy[j]);
                    list.push(alldata[j])
                }
                temp[i]=array;
                alldata1[i]=list;


            }
        }
        console.log(alldata1)
        for(k=0;k<array_temp.length;k++){
            var maxlist=[]
            for(i=0;i<alldata[0].length;i++){
                max=alldata1[k][0][i];
                for(j=0;j<alldata1[k].length;j++){
                    if(alldata1[k][j][i]>=max){
                        max=alldata1[k][j][i];

                    }
                }
                maxlist.push(max)

            }
            allmax[k]=maxlist;

        }
        for(k=0;k<array_temp.length;k++){
            var minlist=[]
            for(i=0;i<alldata[0].length;i++){
                min=alldata1[k][0][i];
                for(j=0;j<alldata1[k].length;j++){
                    if(alldata1[k][j][i]<=min){
                        min=alldata1[k][j][i];

                    }
                }
                minlist.push(min)

            }
            allmin[k]=minlist;

        }
        for(k=0;k<array_temp.length;k++){
            var midlist=[];
            for(i=0;i<alldata[0].length;i++){
                mid=0;
                for(j=0;j<alldata1[k].length;j++){
                    mid=mid+alldata1[k][j][i];
                }
                mid=mid/(alldata1[k].length);
                midlist.push(mid)

            }
            allmid[k]=midlist;
        }
        console.log(allmid)
        var series=[];
        for(i=0;i<temp.length;i++){
            series.push({
                name:array_temp[i],
                symbolSize: 15,
                data: temp[i],
                type: 'scatter'
            })
        }
        var data_temp=[];

        for(i=0;i<array_temp.length;i++){
            data_temp[i]=String(array_temp[i]);
        }

        option = {
            toolbox: {
                orient: 'vertical',
                show : true,
                feature : {
                    mark : {show: true},
                    saveAsImage : {
                        show: true,
                        pixelRatio: 1,
                        title : '保存为图片',
                        type : 'png',
                        lang : ['点击保存']
                    }
                }
            },
            title:{
            },
            legend: {
                left: 'center'
            },
            xAxis: {},
            yAxis: {},
            tooltip: {
                formatter: function(params) {
                    n=params.seriesIndex;
                    var res = '所选类别:'+params.seriesName +'<br/>';
                    res+='样本数目:'+alldata1[n].length+'<br/>';
                    var i ,j;
                    // for(i=0;i<alldata.length;i++){
                    //     if(allxy[i][0]==params.value[0]&&allxy[i][1]==params.value[1]){
                    //         break;
                    //     }
                    // }
                    for(j=0;j<alldata[0].length;j++){
                        res+=alltitle[j]+'的取值范围:'+allmin[n][j]+'--'+allmax[n][j];
                        res+='其均值为:'+allmid[n][j].toFixed(2)+'<br/>';
                    }

                    return res
                }
            },
            series: series
        };

        var Echart = echarts.init(document.getElementById('alg1Echart'));
        Echart.setOption(option);
        // Echart.on('click', function (params) {
        //     // 控制台打印数据的名称
        //
        //     console.log(params);
        // });
    }

    function getAlgSecond(eCode,alg2) {
        $.ajax({
            type:"get",
            url:"${pageContext.request.contextPath}/AutoMlClassifierSelection/ClassifierModel",
            data:{eCode:eCode,alg:alg2},
            dataType:"json",
            beforeSend:function(){
                $("#TabContent2").html("正在处理，请稍后···");
            },
            success:function (result) {
                $("#TabContent2").hide();

                var feature_names=result['feature_names'];
                var data=result['data'];
                var result_label=result['result_label'];
                var xy=result['xy'];

                getAlg2DataTable(feature_names,data,result_label);
                alg2Echart(result_label,xy,data,feature_names)
            },
            complete: function () {
                reloadpage();
            }
        })
    }

    function getAlg2DataTable(feature_names,data,result_label) {
        var tableTitle=[];
        var tabledata=[];
        var title = [];
        var i,j;
        for(i=0;i<feature_names.length;i++){
            title[i] = feature_names[i];
            tableTitle[i]={
                class: 'w100',
                field:''+feature_names[i],
                title:''+feature_names[i]
            };
        }
        title[i] = 'result';
        tableTitle[i] = {
            class: 'w100',
            field:'result',
            title:'result'
        };

        for(i=0;i<data.length;i++) {
            var temp = {};
            for (j = 0; j < data[i].length; j++) {
                temp[title[j]] = data[i][j];
            }
            temp[title[j]] = result_label[i];
            tabledata[i] = temp;
        }

        $('#alg2DataTable').bootstrapTable('destroy');
        $('#alg2DataTable').bootstrapTable({
            icons: {export: 'glyphicon-export'},
            showExport: true,
            // showRefresh: true,
            exportTypes:['csv', 'txt', 'sql', 'doc', 'excel', 'xlsx', 'pdf'],
            exportDataType: "all",
            exportOptions:{
                ignoreColumn: [0,1],  //忽略某一列的索引
                // filename: "分类结果",
                excelstyles: ['background-color', 'color', 'font-size', 'font-weight']
                // onMsoNumberFormat: DoOnMsoNumberFormat
            },
            pagination: true,
            pageSize:"15",
            columns:tableTitle,
            data: tabledata
        });
    }

    function alg2Echart(result_label,xy,data,title){
        var alldata = data;
        var alldata1=[]
        var allmax=[];
        var allmin=[];
        var allmid=[];
        var allxy = xy;
        var alltitle = title;
        var temp=[],i,j,k,n,temp1=[];
        var set_temp=new Set(result_label);
        var array_temp=Array.from(set_temp);

        for(i=0;i<array_temp.length;i++){
            var array=[];
            var list=[];
            for(j=0;j<result_label.length;j++){
                if(result_label[j]==array_temp[i]){
                    array.push(xy[j]);
                    list.push(alldata[j])
                }
                temp[i]=array;
                alldata1[i]=list;


            }
        }
        console.log(alldata1)
        for(k=0;k<array_temp.length;k++){
            var maxlist=[]
            for(i=0;i<alldata[0].length;i++){
                max=alldata1[k][0][i];
                for(j=0;j<alldata1[k].length;j++){
                    if(alldata1[k][j][i]>=max){
                        max=alldata1[k][j][i];

                    }
                }
                maxlist.push(max)

            }
            allmax[k]=maxlist;

        }
        for(k=0;k<array_temp.length;k++){
            var minlist=[]
            for(i=0;i<alldata[0].length;i++){
                min=alldata1[k][0][i];
                for(j=0;j<alldata1[k].length;j++){
                    if(alldata1[k][j][i]<=min){
                        min=alldata1[k][j][i];

                    }
                }
                minlist.push(min)

            }
            allmin[k]=minlist;

        }
        for(k=0;k<array_temp.length;k++){
            var midlist=[];
            for(i=0;i<alldata[0].length;i++){
                mid=0;
                for(j=0;j<alldata1[k].length;j++){
                    mid=mid+alldata1[k][j][i];
                }
                mid=mid/(alldata1[k].length);
                midlist.push(mid)

            }
            allmid[k]=midlist;
        }
        console.log(allmid)
        var series=[];
        for(i=0;i<temp.length;i++){
            series.push({
                name:array_temp[i],
                symbolSize: 15,
                data: temp[i],
                type: 'scatter'
            })
        }
        var data_temp=[];

        for(i=0;i<array_temp.length;i++){
            data_temp[i]=String(array_temp[i]);
        }

        option = {
            toolbox: {
                orient: 'vertical',
                show : true,
                feature : {
                    mark : {show: true},
                    saveAsImage : {
                        show: true,
                        pixelRatio: 1,
                        title : '保存为图片',
                        type : 'png',
                        lang : ['点击保存']
                    }
                }
            },
            title:{
            },
            legend: {
                left: 'center'
            },
            xAxis: {},
            yAxis: {},
            tooltip: {
                formatter: function(params) {
                    n=params.seriesIndex;
                    var res = '所选类别:'+params.seriesName +'<br/>';
                    res+='样本数目:'+alldata1[n].length+'<br/>';
                    var i ,j;
                    // for(i=0;i<alldata.length;i++){
                    //     if(allxy[i][0]==params.value[0]&&allxy[i][1]==params.value[1]){
                    //         break;
                    //     }
                    // }
                    for(j=0;j<alldata[0].length;j++){
                        res+=alltitle[j]+'的取值范围:'+allmin[n][j]+'--'+allmax[n][j];
                        res+='其均值为:'+allmid[n][j].toFixed(2)+'<br/>';
                    }

                    return res
                }
            },
            series: series
        };

        var Echart = echarts.init(document.getElementById('alg2Echart'));
        Echart.setOption(option);
    }

    function getAlgThree(eCode,alg3) {
        $.ajax({
            type:"get",
            url:"${pageContext.request.contextPath}/AutoMlClassifierSelection/ClassifierModel",
            data:{eCode:eCode,alg:alg3},
            dataType:"json",
            beforeSend:function(){
                $("#TabContent3").html("正在处理，请稍后···");
            },
            success:function (result) {
                $("#TabContent3").hide();

                var feature_names=result['feature_names'];
                var data=result['data'];
                var result_label=result['result_label'];
                var xy=result['xy'];

                getAlg3DataTable(feature_names,data,result_label);
                alg3Echart(result_label,xy,data,feature_names)
            },
            complete: function () {
                reloadpage();
            }
        })
    }


    function getAlg3DataTable(feature_names,data,result_label) {
        var tableTitle=[];
        var tabledata=[];
        var title = [];
        var i,j;
        for(i=0;i<feature_names.length;i++){
            title[i] = feature_names[i];
            tableTitle[i]={
                class: 'w100',
                field:''+feature_names[i],
                title:''+feature_names[i]
            };
        }
        title[i] = 'result';
        tableTitle[i] = {
            class: 'w100',
            field:'result',
            title:'result'
        };

        for(i=0;i<data.length;i++) {
            var temp = {};
            for (j = 0; j < data[i].length; j++) {
                temp[title[j]] = data[i][j];
            }
            temp[title[j]] = result_label[i];
            tabledata[i] = temp;
        }

        $('#alg3DataTable').bootstrapTable('destroy');
        $('#alg3DataTable').bootstrapTable({
            icons: {export: 'glyphicon-export'},
            showExport: true,
            // showRefresh: true,
            exportTypes:['csv', 'txt', 'sql', 'doc', 'excel', 'xlsx', 'pdf'],
            exportDataType: "all",
            exportOptions:{
                ignoreColumn: [0,1],  //忽略某一列的索引
                // filename: "分类结果",
                excelstyles: ['background-color', 'color', 'font-size', 'font-weight']
                // onMsoNumberFormat: DoOnMsoNumberFormat
            },
            pagination: true,
            pageSize:"15",
            columns:tableTitle,
            data: tabledata
        });
    }

    function alg3Echart(result_label,xy,data,title){
        var alldata = data;
        var alldata1=[]
        var allmax=[];
        var allmin=[];
        var allmid=[];
        var allxy = xy;
        var alltitle = title;
        var temp=[],i,j,k,n,temp1=[];
        var set_temp=new Set(result_label);
        var array_temp=Array.from(set_temp);

        for(i=0;i<array_temp.length;i++){
            var array=[];
            var list=[];
            for(j=0;j<result_label.length;j++){
                if(result_label[j]==array_temp[i]){
                    array.push(xy[j]);
                    list.push(alldata[j])
                }
                temp[i]=array;
                alldata1[i]=list;


            }
        }
        console.log(alldata1)
        for(k=0;k<array_temp.length;k++){
            var maxlist=[]
            for(i=0;i<alldata[0].length;i++){
                max=alldata1[k][0][i];
                for(j=0;j<alldata1[k].length;j++){
                    if(alldata1[k][j][i]>=max){
                        max=alldata1[k][j][i];

                    }
                }
                maxlist.push(max)

            }
            allmax[k]=maxlist;

        }
        for(k=0;k<array_temp.length;k++){
            var minlist=[]
            for(i=0;i<alldata[0].length;i++){
                min=alldata1[k][0][i];
                for(j=0;j<alldata1[k].length;j++){
                    if(alldata1[k][j][i]<=min){
                        min=alldata1[k][j][i];

                    }
                }
                minlist.push(min)

            }
            allmin[k]=minlist;

        }
        for(k=0;k<array_temp.length;k++){
            var midlist=[];
            for(i=0;i<alldata[0].length;i++){
                mid=0;
                for(j=0;j<alldata1[k].length;j++){
                    mid=mid+alldata1[k][j][i];
                }
                mid=mid/(alldata1[k].length);
                midlist.push(mid)

            }
            allmid[k]=midlist;
        }
        console.log(allmid)
        var series=[];
        for(i=0;i<temp.length;i++){
            series.push({
                name:array_temp[i],
                symbolSize: 15,
                data: temp[i],
                type: 'scatter'
            })
        }
        var data_temp=[];

        for(i=0;i<array_temp.length;i++){
            data_temp[i]=String(array_temp[i]);
        }

        option = {
            toolbox: {
                orient: 'vertical',
                show : true,
                feature : {
                    mark : {show: true},
                    saveAsImage : {
                        show: true,
                        pixelRatio: 1,
                        title : '保存为图片',
                        type : 'png',
                        lang : ['点击保存']
                    }
                }
            },
            title:{
            },
            legend: {
                left: 'center'
            },
            xAxis: {},
            yAxis: {},
            tooltip: {
                formatter: function(params) {
                    n=params.seriesIndex;
                    var res = '所选类别:'+params.seriesName +'<br/>';
                    res+='样本数目:'+alldata1[n].length+'<br/>';
                    var i ,j;
                    // for(i=0;i<alldata.length;i++){
                    //     if(allxy[i][0]==params.value[0]&&allxy[i][1]==params.value[1]){
                    //         break;
                    //     }
                    // }
                    for(j=0;j<alldata[0].length;j++){
                        res+=alltitle[j]+'的取值范围:'+allmin[n][j]+'--'+allmax[n][j];
                        res+='其均值为:'+allmid[n][j].toFixed(3)+'<br/>';
                    }

                    return res
                }
            },
            series: series
        };

        var Echart = echarts.init(document.getElementById('alg3Echart'));
        Echart.setOption(option);
    }
    
    function reloadpage() {
        setTimeout('refresh()', 5000);
    }
    function refresh(){
        url = location.href;
        console.log(url);
        var once = url.split("#");
        if (once[1] != 1) {
            url += "#1";
            self.location.replace(url);
            window.location.reload();
        }
    }

</script>
</html>
