# DS-group2work
Group work for Fundamental of data science

## pre-process  
1.合并三个文件为1个，添加source列来识别来自哪个branch  
2.性别列转化为数值型： 男性为1，女性为0  
3.删除缺失值，重复行只保留一个  
4.删除Salary列异常值1e+21  
5.每次处理后的数据将保存到output文件夹中，文件名为output.csv，并自动编号以区分不同版本。

6.Score列大于850的数量有4个，删掉  
7. 增加每一列的分布图用来识别异常值  

Create a new environment or whatever you want, install the packages in the requirements.txt file by the command:
```
pip install -r requirements.txt
```


