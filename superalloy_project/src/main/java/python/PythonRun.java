package python;
import dao.DataDao;
import dao.FilterDao;
import entity.*;
import org.bson.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import service.SysAccountService;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;
@Service
public class PythonRun {
	@Autowired
	private  DataDao dataDao;
	@Autowired
	private SysAccountService sysAccountService;
	@Autowired
	private FilterDao filterDao;
	@Value("${python.python}")
	private String pythonPath;
	@Value("${file.data}")
	private String filePath;
	@Value("${python.feature-selection}")
	private String featurePath;

	/**该runpython方法针对特征分析的第一层，在其之后还有一个重载的runpython方法，针对的是第二、三、四层
	 *@Description:
	 *@Param: [username, filename, algorithm]
	 *@Return: void
	 */
	public void runpython(String username,String filename,String algorithm){

		try {
			System.out.println("-------------");
			System.out.println("start;;;");
			String fileAbsolutepath=filePath+username+"\\"+filename;//需进行
			// 特征分析的文件(excel或csv格式的文件)的绝对路径
			String[] args = new String[] { pythonPath,
					featurePath+algorithm
					,fileAbsolutepath
					,username
					};
			Process pr = Runtime.getRuntime().exec(args);
			BufferedReader in = new BufferedReader(new InputStreamReader(pr.getInputStream()));
			String line;
			while ((line = in.readLine()) != null) {
				System.out.println(line);
			}
			in.close();
			pr.waitFor();
			System.out.println("end");
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	/**该runpython方法主要是针对特征分析的第二、三、四层
	 *@Description:
	 *@Param: [username, filename, algorithm, expert_remain]
	 *@Return: void
	 */
	public void runpython(String username,String filename,String algorithm,String expert_remain){

		try {
			System.out.println("start;;;");
			String fileAbsolutepath=filePath+username+"\\"+filename;//需进行
			// 特征分析的文件(excel或csv格式的文件)的绝对路径
			String[] args = new String[] { pythonPath,
					featurePath+algorithm
					,fileAbsolutepath
					,username
					,expert_remain};
			Process pr = Runtime.getRuntime().exec(args);
			BufferedReader in = new BufferedReader(new InputStreamReader(pr.getInputStream()));
			String line;
			while ((line = in.readLine()) != null) {
				System.out.println(line);
			}
			in.close();
			pr.waitFor();
			System.out.println("end");
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	/**
	 * testpython包Pythondemo测试该方法
	 * 该方法传递filepath参数到python算法中
	 */
	public void runpython(){
		try {
//			SysAccount account=sysAccountService.getLoginAccount();
			System.out.println("start;;;");
			String filepath="F:\\研究生课程学习资料\\Reinforcement\\RL-project\\glass.csv";
			String username="wjm";
			String expert_importance="1,1,1,0,0.5,1,1";
			String[] args = new String[] { pythonPath,
				"F:\\研究生课程学习资料\\Reinforcement\\RL-project\\feature1.py",filepath,username,"3,6"};
			Process pr = Runtime.getRuntime().exec(args);
			BufferedReader in = new BufferedReader(new InputStreamReader(pr.getInputStream()));
			String line;
			while ((line = in.readLine()) != null) {
				System.out.println(line);
			}
			in.close();
			pr.waitFor();
			System.out.println("end");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public List<String> getHeaderByECode(String eCode){
		SysAccount account=sysAccountService.getLoginAccount();
		
//		List<String> list=dataDao.getDataHeader(account.getUsername(),record);
	 List<String> list=dataDao.getDataHeaderByECode(eCode);

		list.remove("_id");
		list.remove("username");
		list.remove("record");
		list.remove("eCode_");

		return list;
	}

	public List<String> getHeader(String record){
		SysAccount account=sysAccountService.getLoginAccount();

		List<String> list=dataDao.getDataHeader(account.getUsername(),record);
//		List<String> list=dataDao.getDataHeaderByECode(eCode);

		list.remove("_id");
		list.remove("username");
		list.remove("record");
		list.remove("eCode_");

		return list;
	}

}
