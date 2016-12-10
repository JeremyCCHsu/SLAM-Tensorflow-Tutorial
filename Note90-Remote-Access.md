
# Tensorboard

## 從你在實驗室的個人電腦存取 Server 端的 Tensorboard 服務  
我們整棟樓使用的是同一個網域。  
因此，你可以在 server 執行這段指令：  
```shell
tensorboard --logdir=your_model_dir
```
之後，從自己的電腦執行瀏覽器，輸入 Server 的 IP和 Tensorboard 的 port (6006)  
你就可以從自己的電腦瀏覽 Tensorboard 的結果了。  

\*會需要這樣做，是因為 server 通常沒有 GUI。
<br>
<br>

## 從家裡存取 Server端的 Tensorboard 服務
家裡和公司當然不是同一個網域。  
但是，如果我們能 ssh 到公司的 server  
當然也就有可能把 Tensorboard 的訊息傳回來家裡的電腦  
以便透過瀏覽器來看 training 的訊息
方法是

### OS: Linux 或其他 Unix系
```shell
ssh -NfL [隨便自訂的port number]:[Tensorboard的hostname和 port] [server IP]
```

e.g.  
```
ssh -NfL 5566:192.168.1.102:80 user@140.XXX.XXX.XXX  
```
意思是：  
Tunneling from localhost:5566 to 192.168.1.102:80 via user@140.XXX.XXX.XXX

本例中：

1. user 是你在 server 的帳號
2. 140.XXX.XXX.XXX 是 server 的 IP

<br>

#### 指令釋義
> 我在 localhost；我開啟一個傳送門叫 [5566]()  
> 我知道 [192.168.1.102:80]() 那邊有提供 **Tensorboard** 的服務，所以  
> 我先走地下隧道去 user@XXX.XXX.XXX，  
> 再從那邊連結到 [192.168.1.102:80]()  

用個例子來說明：  
> 我在 [家裡的電腦]；我在 [家裡的電腦] 上召喚 localhost:5566 傳送門。  
> 然後，我先打通前往 \[實驗室 server\](user@140.XXX.XXX) 的地下隧道  
> 然後在 [實驗室 server] 打通一條去 \[目標伺服器\](192.168.1.102:80) 的隧道  
> 於是，我便可以透過這些隧道來存取 Tensorboard 的服務

也就是說，你透過「能用 ssh 連線到 server」的事實  
把 server 當作跳板，來存取該網域內的服務。


Note:

0. -Nf 不一定要打。視你的需求而定。
1. -N: 連線後不執行指令
2. -f: 連線後在背景執行
3. -L: 打通隧道 (就上面講的那樣)
4. 如果你連線到 server 的 port 不是預設的 22 而是 80  
   請在行末加上 -p 80
5. 同理，你也可以在 server 上開 Jupyter，讓你在別的地方使用  
(如此一來，家裡或實驗室電腦就可以不用安裝 Python/Jupyter/Tensorflow 了)

<br>

### OS: Windows
這要用到 PuTTY 的進階功能：

1. 在 PuTTY 第一個畫面輸入 server 的IP
2. 展開左方的 SSH，選 Tunnels
3. 在 Source port 隨便填入一個數字 (ex: 5566)
4. 在 Destination 輸入 Server 的IP 和 Tensorboard 的 port (6006)  
   e.g. 140.XXX.XXX.XXX:6006
5. 按 Destination 上方的 Add
6. 按下方的 Open

<br>

### Epilogou
無論是哪個方法，連線後他都會要你輸入帳密  
成功之後會，會跟平常一樣，開啟一個 session。  
這時你就可以不要管這個 session 了。  
打開你的 Chrome/Edge/Firefox/Safari 吧  
在網址列輸入 localhost:5566  
你就可以看 Tensorboard 了 :thumbsup:


## 從你在實驗室的個人電腦存取 Server 端的 Jupyter 服務  
如果先 VPN 到實驗室的桌機的話，我確實可以用上述的方法遠端存取 Jupyter  
(當然，這是說 server 端的 Jupyter 設定都完成的情況)  
但很遺憾，如果沒聯 VPN，我就沒有成功連上 Jupyter server 過了。  


### Reference
1. [利用SSH Tunnel連線至內部網路](http://gwokae.mewggle.com/wordpress/2010/08/%E5%88%A9%E7%94%A8ssh-tunnel%E9%80%A3%E7%B7%9A%E8%87%B3%E5%85%A7%E9%83%A8%E7%B6%B2%E8%B7%AF/)
0. [我所不知道的 SSH 用法](http://chimerhapsody.blogspot.tw/2015/09/ssh.html)
2. [SSH Tunnel 一般场景用法](http://blog.csdn.net/wxqee/article/details/49234595)
3. [SSH原理与运用（二）：远程操作与端口转发](http://www.ruanyifeng.com/blog/2011/12/ssh_port_forwarding.html)
4. [The Black Magic Of SSH: SSH Can Do That?](https://vimeo.com/54505525)

### Miscellaneous Resources
1. [Running Jupyter Notebook Server](http://jupyter-notebook.readthedocs.io/en/latest/public_server.html#notebook-public-server)
