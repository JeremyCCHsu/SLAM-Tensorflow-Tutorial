# OpenAI's Gym for RL
If you want to install on your own device, go to Sec. I.  
If you plan to run Gym on a Docker in the server, go to Sec II.  
<br/>
<br/>

## Section 0: My Working Environment
|      | OS           | IP                               |
|------|--------------|----------------------------------|
|Local | Windows 7    | XXX.XXX.19.81                    |
|Server| Ubuntu 16.04 | XXX.XXX.22.106                   |
|Docker| Ubuntu       | ???.???.???.??? (doesn't matter) |
<br/>
<br/>


## Section I: [Official Installation](https://gym.openai.com/docs)
安裝比想像中困難。  
Gym 還需要 OpenGL等等的 dependencies  
我自已因為都是透過 server 在跑我的 code  
而我並沒有 server 的權限，沒辦法裝這些  
另一個難點是： server 也沒有 xvfb 之類的東西，所以我根本沒有辦法看 Gym code 吐出來的影片。  
所以後來我是用 docker，這會有其他更多的困難產生，但可以解決。  
<br/>
<br/>


## Section II: Gym on a Docker: using [a pre-built image](https://hub.docker.com/r/eboraas/openai-gym/)
這個 image 讓你可以在 Jupyter Notebook 中，interactively 操作 Gym code  
(雖然他有個缺點，就是 Jupyter 不會顯示出動畫；這我們後面會解決)  
推薦這個 Docker repo 是因為他已經搞定了 Tensorflow、Jupyter、Gym  
只要在 server 上執行：
```bash
docker run -d -p 8888:8888 -p 6006:6006 -v /path/to/notebooks/:/mnt/notebooks/ eboraas/openai-gym
```
然後透過瀏覽器連到 localhost:8888  
就可以進入 Jupyter 開始玩 Gym 囉~  
(註: docker run 沒找到 image 的時候，會自己到他的 repository 裡面找；所以這邊才會連安裝的手續都沒看到。)
<br/>
<br/>


## Section III: Viewing Gym's Output Videos
上一節用 Jupyter 來玩 Gym 沒辦法顯示項官方文件上面那些影片 (e.g. CartPole 左右移動的那種)，因此我們必須訴諸 VNC 來解決這個問題。  

- [Build a Docker Image](#build-a-docker-image)
- [Configure Putty and the Server](configure-putty-and-the-server)
- [Install VNC Viewer](#install-vnc-viewer)
- [Run Your Container](#run-your-container)

<br/>


### Build a Docker Image
首先，我們建立一個 Docker image  
```bash
mkdir gymvnc
cd gymvnc
```

編輯一個空白文件，名曰 Dockerfile (首字大寫，無副檔名)  
```
# OpenAI Gym over VNC
FROM eboraas/openai-gym
RUN apt-get update
RUN apt-get install -y x11vnc xvfb
RUN mkdir /.vnc
RUN x11vnc -storepasswd 9487 /.vnc/passwd
```
(請參考資料夾 Note91-OpenAI-Gym/)  

接著，建立一個 docker image
```bash
docker build -t gymvnc .
```
<br/>


### Configure Putty and the Server

* Putty  

   1. 設定 X11 forwarding:  
   **Connection > SSH > X11 > X11forwarding**  
   Activate the checkbox: Enable X11 forwarding  
   3. 設定 port forwarding:  
   **Connection > SSH > Tunnels**  
     Source port: 5900 *(這是你本機端的 port)*  
     Destination: [server IP]:5900 *(這好像是 x11vnc 預設的 port)*

* Server Environment Variables  
   ~$ `export DISPLAY=:0`  

<br/>


### Install VNC Viewer
安裝 VNC Viewer (我是安裝其 Chrome App)  
<br/>


### Run Your Container
全都完成之後，從 server 啟動 docker container:  
```bash
docker run -p 5900:5900 -e HOME=/ -v [dir-to-your-script]:/mnt/notebooks/ gymvnc x11vnc -forever -create -usepw
```
記得把上面 [dir-to-your-script] 改成你放你要跑的 Python script 的地方。  


其中  
> 1. `-p 5900:5900` 是 port forwarding，用來打通 server 和 container 的 port
> 2. `-e HOME=/` 這是環境變數 (我不確定其重要性...)  
> 3. `x11vnc -forever -usepw -create` 是在 container 中真正執行的程式  
>    我們必須執行 x11vnc，才能把 container 的畫面傳回來 local 端
> 4. `-v [path on the server]:[path in the container]` 
>    是用來打通 server 和 container 的硬碟的。
>    這樣做的原因是因為 docker container 裡面做過的所有事情都會消失，
>    除非你有把真實的硬碟載上去，否則無法保留結果；
>    另外，如果要把你的程式傳進去，也必須用到這個功能。
> 5. 我的範例程式放在 `~/proj/gvncpy` 中。
>    但是 docker 要求 argument 必須是絕對路徑。  


最後就是在本機端用 VNC Viewer 連線到 localhost:5900  
> 本例中有設定密碼為 9487。  
> 若不要密碼的話，`docker run` 那邊，
> `x11vnc` 後面不要加 `-usepw`就好了  

之後，你就會看到一個 Terminal，路徑在`/mnt/notebooks/`。  
執行`python test.py`即可看到結果。   


我執行的範例是 Gym 官網上的 CartPole (with some modification):  
```python
import gym
env = gym.make('CartPole-v0')
env.reset()

counter = 0
for _ in range(1000):
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    if done:
        #print('Game failed')
        counter += 1
        print('Game failed', counter)
        env.reset()
```
(請參考資料夾 Note91-OpenAI-Gym-test/)
<br/>
<br/>


## Final Remarks  
我有點意外的是: Matplotlib 竟然沒有辦法透過這個方法顯示。  
我有看到 Stackflow 上有人說要用 GTK backend  
但我沒有深究這個問題，因為我向來是用 plt.savefig 把圖存下來，傳回本機端看。
<br/>
<br/>


## Reference
Most Helpful:  
- [Can You Run GUI Apps in a Docker Container](http://stackoverflow.com/questions/16296753/can-you-run-gui-apps-in-a-docker-container)

Related:
-[Docker Desktop: Your Desktop over SSH Running inside of a Docker Container](https://blog.docker.com/2013/07/docker-desktop-your-desktop-over-ssh-running-inside-of-a-docker-container/)
- [Running GUI Apps with Docker](http://fabiorehm.com/blog/2014/09/11/running-gui-apps-with-docker/)
- [Running a GUI Application in a Docker Container](https://linuxmeerkat.wordpress.com/2014/10/17/running-a-gui-application-in-a-docker-container/)
- [安裝 XVFB 做 Selenium 測試](https://www.puritys.me/docs-blog/article-262-%E5%AE%89%E8%A3%9D-XVFB-%E5%81%9A-Selenium-%E6%B8%AC%E8%A9%A6.html)