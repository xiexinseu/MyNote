# ubuntu共享

Windows 可以通过samba账号密码访问 Ubuntu 設置的共享目录。



设置账号密码和重置密码的方法如下：

```bash
sudo smbpasswd -a user  
New SMB password:  
Retype new SMB password:  
```

以下是 使账号 champwang 失效：

```bash
sudo smbpasswd -d champwang  
[sudo] password for user:   
Disabled user champwang.  
```

去除添加的账号：

```bash
sudo smbpasswd -x champwang  
Deleted user champwang.  
```

以下是 smbpasswd 的說明：

```bash
$ smbpasswd -h  
When run by root:  
    smbpasswd [options] [username]  
otherwise:  
    smbpasswd [options]  
  
options:  
  -L                   local mode (must be first option)  
  -h                   print this usage message  
  -s                   use stdin for password prompt  
  -c smb.conf file     Use the given path to the smb.conf file  
  -D LEVEL             debug level  
  -r MACHINE           remote machine  
  -U USER              remote username  
extra options when run by root or in local mode:  
  -a                   add user  
  -d                   disable user  
  -e                   enable user  
  -i                   interdomain trust account  
  -m                   machine trust account  
  -n                   set no password  
  -W                   use stdin ldap admin password  
  -w PASSWORD          ldap admin password  
  -x                   delete user  
  -R ORDER             name resolve order  
```

另外，介紹下 ubuntu 系統给 windlows 共享文件夹方法：

选中要共享的文件夹，

右键选properties, 

在share 一栏，勾上"share this folder", 

点击 Modify,

就可以进行文件夹的共享了。