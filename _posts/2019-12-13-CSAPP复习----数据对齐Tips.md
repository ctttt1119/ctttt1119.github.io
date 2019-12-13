---
layout:     post
title:      CSAPP复习
subtitle:   数据对齐 + switch
date:       2019-12-13
author:     Shawn
header-img: img/home-bg-o.jpg
catalog: true
tags:
    - CSAPP期末复习










---

# CSAPP复习：数据对齐 + switch

## 一、数据对齐

1. 这里首先注意一个不同点：
   在linux-32机器中，8字节的数据之需要对其地址为4的倍数即可，但是在x86-64中规则更加严格，8字节数据必须对其地址为8的倍数。

2. 要产生一个指向结构内部对象的指针，我们只需将结构的地址加上该字段的偏移量，使用的指令为lea，例如：

   ```c
   struct ms_pacman { 
   short wire; 
   int resistor; 
   union transistor { 
   	char bjt; 
   	int *mosfet; 
   	long vacuum_tube[2];
   } transistor; 
   struct ms_pacman *connector;
   };
   char* inky(struct ms_pacman *ptr) { 
       return &(ptr->transistor.bjt);
   }
   ```

   其对应的汇编如下：

   ```c
   lea 0x8(%rdi),%rax retq
   ```

3. 在对于结构中嵌入了结构指针的，将他当作一个指针按照8字节计算即可

4. 对于结构中有联合的，计算偏移量时需注意，用以下例子进行讲解：

   ```c
   struct ms_pacman { 
   short wire; 
   int resistor; 
   union transistor { 
   	char bjt; 
   	int *mosfet; 
   	long vacuum_tube[2];
   } transistor; 
   struct ms_pacman *connector;
   };
   ```

   在这个例子中，结构中存在一个联合，首先short占2字节，int占4字节，为了对其，int从4开始，所以联合从8开始，这里的联合总大小是16字节，但是并不需要16字节对其，因为对其需要看联合内部的数据，联合内部数据分别为1字节、8字节，所以只需要8字节对其即可，这里正好是从8字节开始，所以不需要再额外增加偏移量。

-----

## 二、switch

书本p.159

书上写的很清楚了,贴一个例题:

**Consider the following C code and assembly code:**

```c
int lol(int a, int b) {
switch (a) { 
case 210: b *= 13; __break;___
case 213: b = 18243; _____
case 214: b *= b; __break;___
case 216: 
case 218: 
        b -= a; __break;___
case 219: b += 13; _____
default: b -= 9;
} return b;
}
```

![image.png](https://i.loli.net/2019/12/13/O7qaAThBndkSeXp.png)

Using the available information, fill in the jump table below. (Feel free to omit leading zeros.) Also, for each case in the switch block which should have a break, write break on the corresponding blank line.

![image.png](https://i.loli.net/2019/12/13/eyDXF5dPL132MV9.png)

