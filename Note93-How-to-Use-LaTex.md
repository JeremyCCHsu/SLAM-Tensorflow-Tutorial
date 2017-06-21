
# Latex文章範本
https://www.overleaf.com/10001186xqfdpvtmygpx#/36769147

# 幾個超推薦的重點
1. 不必管理字型  
   例如如果小節標題要大寫，LATEX會自動做
2. 不必手動改排版
2. 不必一直修改 reference (LATEX會依照出現順序自動編號)
3. 式子比較好寫 (尤其當你突然想把 G 全部換成 A 的時候...)
4. 方程式自動編號
7. 方程式的字型可以很漂亮
5. 任何東西(節、圖、表、方程式都可以用 `\label` 來自動編號)
6. 太長的字會自動用連字號切斷，不會像 WORD 那樣有很醜的縮排

# 幾大要點說明

## 基本使用
其實就跟平常寫文章是一樣的，反正就寫。  
需要注意的只有：
1. 空一行表示分段
2. 空格通常沒有意義。就算一句寫完空三格再寫下一句，最後的 PDF 也只會空一格。
3. 方程式中，如果需要空格，請使用 `~`
3. 承上，所以我通常一行一句。
3. 用 `%` 來寫註解，或者把不要的段落註解掉 (快捷鍵 `Ctrl+/`)

## 常用指令
所有特殊指令都是以 `\` 開頭；你也可以自己定義特殊指令。

### 一般指令
最常用到的就是 `\section{X}`，他會把大括號中的文字變成節標題。
他的兒子 `\subsection{}` 則是小節標題、  
孫子 `\subsubsection{}` 則是在下一層的標題。  
我不確定是否可以無限 `sub` 下去，但我建議不要寫這麼多縮排。

### 成對指令
`\begin{X}` 和 `end{X}` 必定成對。
這兩個指令在寫下列東西的時候會使用到。  
  1. 圖(X = figure)、
  2. 表(X = tabular)、
  3. 方程式(X = equation)  
  4. 整份論文 (X = document；但我們通常不會動到這個) 

例子請看範例文件。


### 自定義指令
你可以使用 `\newcommand{}{}`來定義新指令。
這在定義方程式變數的時候很有用。  

#### 一般用法 
Syntax: `\newcommand{指令縮寫} {實際形式}`  
(兩括號間有沒有空格沒差)  
例如：  
`\newcommand{\A} {A_m}`  

在書寫的時候，你會希望變數名稱越簡單越好；  
所以，就算是「A下標m」，撰寫的時候如果只需要打「\A」，豈不是爽爆了?

#### 高級用法  
Syntax: `\newcommand{指令縮寫}[變數數量] {實際形式}`
例如：  
`\newcommand{\E}[2] {\mathbb{E}_{#1}\big[ #2 \big]}`
這是定義一個「有大一點的中括號的期望值符號」。  

#### Overriding (推翻之前定義過的符號)
如果你要定義某個已經被定義過的變數(也就是內建的變數)，請使用 `renewcommand`
e.g.  
```tex
\renewcommand{\xi} {x_i}
```



## 參考資料
要引用別人的文章前，你需要編輯一個 `*.bib` 檔。  
你可以把所有的文章(就算這次沒有 cite 到的) 都放進去。  
反正 LATEX 只會列出你這篇文章有用到的那些。  

加入 reference 的方法，是在文章的最末 (`\end{document}`) 寫以下兩行
```tex
% syntax
\bibliographystyle{格式}
\bibliography{文獻集檔案}
```

其中，`格式` 要改成你要投搞得地方的格式 (APSIPA 應該是走 IEEE 格式，所以這次不用改)。  
`文獻集檔案` 則是填文獻集的檔名(不含附檔名)。  
接下來，我們就來看文獻檔要怎麼編纂。

### 編輯參考文獻檔 (本例中的 `reference.bib`)
通常呢，我都是到 [dblp](http://dblp.uni-trier.de) 去找我要的文章  
找到之後，在標題前面有個下載符號，滑鼠移過去，選「export BibTex」
然後把出現的字全部複製貼上到文獻檔中就好了。  


如果找不到，你也可以自己編輯。
你會發現：文獻檔 (`*.bib`) 中的格式很簡單
```bib
@article{DBLP:journals/corr/MohamedL16,
  author    = {Shakir Mohamed and
               Balaji Lakshminarayanan},
  title     = {Learning in Implicit Generative Models},
  journal   = {CoRR},
  volume    = {abs/1610.03483},
  year      = {2016},
  url       = {http://arxiv.org/abs/1610.03483},
  timestamp = {Wed, 02 Nov 2016 09:51:26 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/MohamedL16},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

### 引用 (cite別人的文章)
`\cite{X}`，X 為文章暱稱。  
暱稱就是文件檔中，`@article{`後面出現的字串。  
由於該字串是可以自訂的，所以我稱之為暱稱。  
(但是一般來說，我不會特別去改這串字)

### 參看 (本文中有出現的編號項目)
在會被參考的圖、表、段落下加入 `\label{}` 來做標記 (填入暱稱)。  
之後，在段落中使用 `ref{}` 來引述該標記。  
Note:
  1. 這個功能和「引用」其實很像。  
  2. 暱稱可以是任何字串，所以可以用冒號、等號、加號等等  
  3. 一般會建議依據參考的內容來給予暱稱，例如 `sec:xx, eq:xx, fig:xx, tab:xx`  

e.g.  
```tex
\section{Introduction}
\label{sec:intro}
...


\section{Experiment}
As we have described in Sec. \ref{sec:intro},
```


## 方程式

### 段落中的方程式 (Inline functions)
用成對的錢字號圍起來，例如  
```tex
Note that $w$ is the weight vector.
```

### 單獨成行的方程式
用 begin-end pair 所建立的 equation environment 圍起來，例如
```tex
\begin{equation}
  A = B + C
\end{equation}
```

### 符號自定義
這和前面講的「自定義指令」完全相同。


## 範本裡面已經做了的 (所以你這次不需要做)

### Include packages
Syntax: `\usepackage{}`  
注意這一定要在文章最開頭，不能在文章開始之後才 include 

