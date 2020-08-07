<%--
  Created by IntelliJ IDEA.
  User: buming
  Date: 2019/4/11
  Time: 15:55
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
                <%--<img src="${ctx}/pic/logo_03.png" style="width: 100%; height: auto;max-width: 100%; display: block;padding-top: 10px">--%>
                <img src="${ctx}/pic/logo_03.png" style="width: 100%; height: auto;max-width: 100%; display: block;padding-top: 10px">
            </div>
            <div class="col-sm-7">
                <div style="height: 87px"></div>
                <nav class="nav navbar-nav nav-collapse my-nav" role="navigation" id="Hui-navbar">
                    <ul class="cl" style="margin: 0px;padding: 0px">
                        <li><a href="/"  style=";color: rgb(117,117,117)">首页</a></li>
                        <li><a href="${ctx}/page/dataUpload"  style=";color: rgb(117,117,117)">数据上传</a></li>
                        <%--<li><a href="${ctx}/page/machineLearningList">任务列表</a></li>--%>
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
                <div class="step-content row-fluid position-relative"
                     id="step-container">
                    <div class="step-pane active" id="step1">
                        <div class="panel-group" id="accordion">
                            <div class="panel panel-success">
                                <div class="panel-heading">
                                    <h3 class="panel-title">
                                        元特征展示
                                    </h3>
                                </div>
                                <div id="TabContent" class="tab-content center"></div>
                                <div id="myTabContent" class="tab-content">
                                    <div class="tab-pane fade in active" id="metaFeatures">
                                        <div class="row">
                                            <div class="col-sm-12" class="table-responsive" style="overflow:auto;">
                                                <table id="clusterDataTable" class="table-responsive">

                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div >
                                    <input class=" btn btn-success radius f-r height-auto size-X" type="button" onclick="Recommend_UIM();" value="下一步" >
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

</body>

<script>
    var eCode='<%=request.getParameter("eCode")%>';

    $(function () {
        getMetaFeatures(eCode);
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
                            color:'#3399FF'
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
                            color:'#969696'
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

    function getMetaFeatures(eCode) {
        $.ajax({
            type:"get",
            url:"${pageContext.request.contextPath}/AutoMlClassifierSelection/MetaFeatures",
            data:{eCode:eCode},
            dataType:"json",
            beforeSend:function(){
                $("#TabContent").html("正在处理，请稍后···");
            },
            success:function (result) {
                $("#TabContent").hide();
                var mf_list = result['mf_list'];
                var meta_feature = result['meta_feature'];
                getTable(mf_list,meta_feature);
                // getRecommend_User(eCode);
                // getRecommend_Item(eCode);
                // getRecommend_Model(eCode);
            },
            complete: function () {
                reloadpage();
            }
        })
    }

    function getTable(mf_list,meta_feature) {
        var tableTitle = [];
        var tableData = [];
        var i;
        for(i=0;i<mf_list.length;i++){
            tableTitle[i]={
                class: 'w100',
                field:''+mf_list[i],
                title:''+mf_list[i]
            };
        }

        var temp = {};
        for (i=0;i<meta_feature.length;i++){
            temp[mf_list[i]]=meta_feature[i]
        }
        tableData[0] = temp

        $('#metaFeatures').bootstrapTable('destroy');
        $('#metaFeatures').bootstrapTable({
            pagination: true,
            pageSize:"15",
            columns:tableTitle,
            data: tableData
        });
    }

    function Recommend_UIM() {
        window.location.href="${ctx}/page/Recommend_UIM_classifier?eCode="+eCode;
    }
</script>
</html>