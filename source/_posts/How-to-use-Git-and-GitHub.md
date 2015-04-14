title: "How to use Git and GitHub"
date: 2015-04-13 16:55:26
categories: Engineering
tags: 
---
##Learning Material

* Best thing to start is this tutorial: [廖雪峰的Git教程](http://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)
* Git cheat sheet: [Git Cheat Sheet](https://training.github.com/kit/downloads/github-git-cheat-sheet.pdf)
* Git commands: [jark's blog about Git](http://wuchong.me/blog/2015/03/30/git-useful-skills/)

<!--more-->

![Git Work Flow](http://7xikhz.com1.z0.glb.clouddn.com/81b78497jw1eqnk1bkyaij20e40bpjsm.jpg)


## The most frequent Git Commands: 

**Create repository, copy repository from remote(本地创建 || 远程复制)**

```bash
git init
git clone git@server-name:path/repo-name.git

```
**Check status, log, diff**

```bash
git status
git log
git log --pretty=oneline filename #一行显示
git diff   #check the difference between the current state and previous one
git diff -- readme.txt 

```

**Add Files**

```bash
git add file1.txt
git add file2.txt
git add file3.txt
git commit -m "add three files"

```

**Go to the previous versions**
```bash
git reset --hard HEAD^  # Head is current one, HEAD^ is the previous one, HEAD^^ two step before
git reset --hard id     # id can be viewed use git log

```
**Reset current working directory**

```bash
git checkout -- <file>  #丢弃工作区上某个文件的修改
git reset HEAD <file>   #丢弃暂存区上某个文件的修改，重新放回工作

```
** delete file**
```bash
git rm <file>           #直接删除文件
git rm --cached <file>  #删除文件暂存状态
```

** Add remote repository and push **
```bash
git remote add origin git@server-name:path/repo-name.git  #添加一个远程库
git push -u origin master   #第一次推送master分支的所有内容
git push origin master    #推送到远程master分支(以后)

```

** Branches **
```bash
git branch dev              #只创建分支
git checkout -b dev  #创建并切换到 dev 分支
git branch   # list all the branches
git checkout master   # 切换分支
git branch -d dev    # delete branch
git merge --no-ff develop   #把 develop 合并到 master 分支，no-ff 选项的作用是保留原分支记录
git merge dev  #when in master branch, call this, will merge dev and master
```

**Stash**
```bash
git stash           #储藏当前工作
git stash list      #查看储藏的工作现场
git stash apply     #恢复工作现场，stash内容并不删除, need to use git stash drop 删除
git stash pop       #恢复工作现场，并删除stash内容
```
** Push to remote server**
```bash
git push origin master    #推送到远程master分支
git push origin dev   #dev分支
```

**Pull from remote server**
```bash
git checkout -b dev origin/dev  #创建远程origin的dev分支到本地，并命名为dev
git pull origin master          #从远程分支进行更新 
git fetch origin master         #获取远程分支上的数据
git branch --set-upstream branch-name origin/branch-name#可以建立起本地分支和远程分支的关联，之后可以直接git pull从远程抓取分支。
```
另外，git pull = git fetch + merge to local

**Delete romote branch**
```bash
git push origin --delete bugfix
```
**Tag**
```Bash
git tag         #列出现有标签 
git show <tagname>  #显示标签信息
git tag v0.1    #新建标签，默认位 HEAD
git tag v0.1 cb926e7  #对指定的 commit id 打标签
git tag -a v0.1 -m 'version 0.1 released'   #新建带注释标签

git checkout <tagname>        #切换到标签

git push origin <tagname>     #推送分支到源上
git push origin --tags        #一次性推送全部尚未推送到远程的本地标签

git tag -d <tagname>          #删除标签
git push origin :refs/tags/<tagname>      #删除远程标签

```
** .gitignore ** 
ignore updates in some files
