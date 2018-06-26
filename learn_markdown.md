# MarkDown语法

## 二级标题

MarkDown共六级标题

### 区块引用
> This is the first level of quoting.
>
> > This is nested blockquote.
> > > 
> > > hello
>
> Back to the first level.

### 无序列表

#### 方法一
+ red
+ blue
+ black

#### 方法二
- red
- blue
- black

#### 方法三
* red
* blue
* black

### 有序列表
有序列表生成序数与数字并无实质关联
1. red
2. blue
3. black

### 段内引用
* A list item with a blockquote:
  > This is a blockquote
  > inside a list item.

1986\.  What a great season.

### 分隔线
至少连续三个"+"或三个"-"或三个"*"
* * *
* 
- - -
* 
- - - - - - - - - -

### 代码块
Here is an example of AppleScript:

    tell application "Foo"
        beep
    end tell

### 显示链接

#### 方法一：行内式
- welcome to [UESTC](http://www.uestc.edu.cn)
- welcome to [UESTC](http://www.uestc.edu.cn "电子科技大学")

#### 方法二：参考式
+ welcome to [UESTC] [link_1] reference-style link

[link_1]: http://www.uestc.edu.cn "电子科技大学"

### 隐式链接（掌握）：参考式
- [Google] []

[Google]: http://www.google.com

- I get 10 times more traffic from [Google][] than from
[Yahoo][] or [MSN][].

[google]: http://google.com/    "Google"
[yahoo]:  http://search.yahoo.com/  "Yahoo Search"
[msn]:    http://search.msn.com/    "MSN Search"

### 强调
*single asterisks*
_single underscores_
**double asterisks**
__double underscores__ 
\*hello

### 反引号的使用

#### 生成局部代码块
* Use the `print()` function.
* 在代码区段内插入反引号：`` 插入反引号` ``
* A backtick-delimited string in a code span: `` `foo` ``
----
    `<Hello MarkDown & Sublime Text>`
`<Hello MarkDown>`
    `<Hello MarkDown & Sublime Text>`
    
    `<Hello MarkDown & Sublime Text>`
----

### 图片的使用

#### 方法一：行内式
- Thie is my ![profile](D:/profile.jpg "My profile").

####方法二：参考式
1. This is my ![profile][]
2. This is my computer ![wallpaper][wp]

[profile]: D:/profile.jpg "参考式方法一"
[wp]: D:/wallpaper.jpg "参考式方法二"

### 自动链接
<http://www.google.com>

### 使用"\\"插入特殊字符
\\   反斜线
\`   反引号
\*   星号
\_   底线
\{\}  花括号
\[\]  方括号
\(\)  括弧
\#   井字号
\+   加号
\-   减号
\.   英文句点
\!   惊叹号
