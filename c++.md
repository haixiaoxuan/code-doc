# c++

------

```c++
#include<iostream>
using namespace std;

int main(){
    cout << "hello world" << endl;
    
    // 这行代码的意思是让程序暂停，这样cmd窗口会暂时不退出，方便演示。
    system("pause");
    reuturn 0;
}
```



### g++

```
g++ -v
g++ helloworld.cpp
	-o helloworld.exe		# 指定生成的可执行文件的名称

g++ helloworld1.cpp helloworld2.cpp -o helloworld.exe	# 指定多个源文件生成一个exe

	
```





### 变量

```c++
int a = 10;
cout << "a = " << a << endl;
```





### 常量

​	① 宏常量： #define 常量名 常量值

​	② const修饰的变量：const 数据类型 常量名 = 常量值





### 整型

​	short \int \long(windows 4 byte, linux32 4 byte, linux64 8 byte) \long long

```c++
    short num1 = 10;
    int num2 = 10;
    long num3 = 10;
    long long num4 = 10;

    cout << "num = " << sizeof(short) << endl;
    cout << "num = " << sizeof(int) << endl;
    cout << "num = " << sizeof(long) << endl;
    cout << "num = " << sizeof(long long) << endl;

	// 查看变量类型名称
	cout << typeid(num1).name << endl;
```





### 浮点型

​	float (可以保留七位有效数字包括整数位)、double（15-16位有效数字）

```c++
    float num1 = 3.14f;
    double num2 = 3.14;

    cout << "num = " << num1 << endl;
    cout << "num = " << num2 << endl;   // 默认最长显示六位
```





### 字符型

char ... 

```c++
    char a = 'a';
    cout << "a " << (int)a << endl;
```





### 字符串类型

​	两种风格

```c++
    char str1[] = "hello world";
    cout << str1 << endl;

    string str2 = "hhhh";
    cout << str2 << endl;
```





### 布尔类型

​	真(非零)\ 假(0)

```c++
    bool b = false;
    cout << b << endl;
```





### 输入输出

```c++
    int num = 0;
    cin >> num;
    cout << num << endl;
```



### if语句

```c++
    int num = 0;

    if(num > 10){
        cout << "num > 10" << endl;
    }else if(num < 0){
        cout << "num < 0" << endl;
    }else{
        cout << "hello world" << endl;
    }

```



### 三目运算符

```c++
    cout << (1 == 1 ? 0 : 1) << endl;
```



### switch 语句

```c++
    int num = 10;

    switch (num)
    {
    case 1:
        cout << 1 << endl;
        break;
    
    default:
        cout << "default" << endl;
        break;
    }
```



### 循环语句

```c++

	// while
    while (num < 20)
    {
        cout << num << endl;
        num++;
    }

	// do while ... 

	// for
    for(int i = 0; i < 10; i++){
        cout << i << endl;
    }
```





### 随机数生成

```c++
#include<ctime>    

	// 添加随机数种子
    srand((unsigned int)time(NULL));
    cout << rand() << endl;
```





### 数组

```c++
    int arr[5];
    arr[0] = 0;
    arr[1] = 1; 

    int arr2[5] = {1, 2, 3, 4, 5};

    // 计算数组长度
    cout << sizeof(arr) / sizeof(arr[0]) << endl;

	// 数组作为参数传递时
	void test(int * arr){..}
```





### 函数

```c++
int sum(int a, int b){
    return a + b;
}

// 如果函数定义在函数调用之后，需要在调用之前进行声明， int sum(int a, int b);
```



### 多文件开发

```c++
1. 创建 .h 的头文件
2. 创建 .cpp 的源文件
3. 在头文件中写函数声明
4. 在源文件中写函数定义

--- swap.h
#include <iostream>
using namespace std;

void swap(int a, int b);


--- swap.cpp
#include "swap.h"

void swap(int a, int b){
    int tmp = a;
    a = b;
    b = tmp;

    cout << "a = " << a << endl;
    cout << "b = " << b << endl;
}

--- main
#include<iostream>
using namespace std;
#include "swap.h"



int main(){

    swap(1, 2);

    system("pause");
    return 0;
}
```



### 指针

在32位操作系统下指针占四个字节，64位占八个字节

```c++
    int a = 1;
    int * p = &a;

/*
空指针：指针变量指向内存中编号为0的空间
	用途：初始化指针变量，空指针指向的空间不能被访问
*/
	int * p = NULL;


/*
野指针: 指针变量指向非法的内存空间
*/
	int * p = (int *)0x1100;


/*
const 修饰指针 -- 常量指针
const 修饰常量 -- 指针常量
const 既修饰常量，又修饰指针


常量指针：
	const int * p = &a;
	特点：指针的指向可以修改，但是指针指向的值不可以修改
指针常量
	int * const p = &a;
	特点：指针的指向不可以修改，指针指向的值可以修改

	const int * const p = &a;
	特点: 都不可以修改
*/



// 指针和数组

    int arr[] = {1, 2, 3};

    int * p = arr;
    cout << *p << endl;
    p++;
    cout << *p << endl;
```



### 结构体

​	note: 结构体当参数传递时是值传递

```c++
struct Student{
    string name;
    int age;
};

int main(){

    // struct 关键字可以省略
    struct Student s1;
    s1.name = "xiaoxuan";

    struct Student s2 = {"xiaoxuan2", 18};
    
    // 结构体指针 
    struct Student * p = &s2;
    // 结构体指针通过 -> 访问属性
    p->name;
    p->age;
    
    
    // const 修饰结构体
    // 以这种方式定义的参数，不允许函数中有修改行为
    void test(const struct Student *p){...}
```



### 内存模型

```
代码区：存放函数二进制代码，由操作系统管理
全局区：存放全局变量，静态变量和常量
栈区：由编译器自动分配回收，存放参数值局部变量等
堆区：由程序员分配和释放
```



### new 

```c++
    // 使用new关键字在堆中创建一个值为10的int类型，并将地址返回
    int * p = new int(10);

    // 开辟一个大小为10的int数组
    int * arr = new int[10];

    // 使用delete 释放内存
    delete p;

	// 释放数组
	delete[] arr;
```





### 引用

```c++
    int a = 10;
    
    // 相当于给 a 起别名
    int &b = a;
	cout << b << endl;

	引用必须要初始化
    引用一旦初始化后就不能被修改（所指向内存地址不会改变）
        
        
// 引用当参数传递， 也会改变实参（即非值传递，是引用传递）
void swap(int &a, int &b){
    int tmp = a;
    a = b;
    b = tmp;
}

int main(){

    int a = 10;
    int b = 20;
    swap(a, b);
    cout << a << endl;
    cout << b << endl;
} 




    /*
    	不要返回局部变量的引用
    	函数调用可以写在等号左边
     */
int& test(){
    static int c = 1000;	// 静态变量，不在栈中
    return c;
}

int main(){

    test() = 100;
    int a = test();
    cout << a << endl;

    system("pause");
    return 0;
}


note: 本质上就是指针常量
```



常量引用：

​	在函数形参列表中加 const关键字修饰参数，防止形参改变实参

​	void test(const int &a){...}



### 高级函数

```c++
默认参数：
note:
	func(int a, int b = 10){...}
	func(int a, int b);
	函数定义和函数声明中，只能有一个地方来定义默认值，不然编译器没办法识别。
        
        
占位参数：
    占位参数也可以有默认值
    func(int a, int){...}
	func(10, 10)
        

函数重载：
    note: 函数返回值不同不能作为函数重载的条件
    引用作为重载条件
    函数重载碰到参数默认值
```





### 类

```c++
class Student{
    public:
    int age;
    int getAge(){
        return age;
    }
};


int main(){

    Student s;
    s.age = 100;
    cout << s.getAge() << endl;

    system("pause");
    return 0;
}
```

访问权限：

​	public

​	protected

​	private

struct 和 class 唯一的区别：

​	成员的默认访问权限不一样，struct默认是public，class默认时 private



### 构造函数与析构函数

如果自己不实现，会默认有空实现

```c++
class Person{

    public:
    Person(){
        cout << "construct >> " << endl;
    }

    // 析构函数不能有参数
    ~Person(){
        cout << "delete >> " << endl;
    }
};

int main(){

    Person p;
    // system("pause");
    return 0;
}
```



### 构造函数的分类和调用

```c++
class Person{

    public:
    // 拷贝构造函数
    Person(const Person &p){
        age = p.age;
    }
    // 无参构造函数
    Person(){}
    // 有参构造函数
    Person(int age){}

    int age;
};

int main(){

    /*
        调用方式： 括号法，显示法，隐式转换法
     */

    // 1 括号法
    Person p1;
    Person p2(10);
    Person p3(p1);
    // Person p4();    // 这种情况不会生成对象，编译器会把他当作函数声明

    // 2 显示法
    Person p5;
    Person p6 = Person(10);
    Person p7 = Person(p5);
    // Person(10);      匿名对象，当前行执行完成之后会马上被回收

    // 3 隐式转换法
    Person p8 = 10;     // 等价于 Person p8 = Person(10);
    Person p9 = p8;


    // system("pause");
    return 0;
}


调用函数时，值传递的方式会触发调用拷贝构造函数；
函数返回时，会将值进行拷贝返回，也会触发拷贝构造函数
```

默认情况下，c++编译器至少给一个类添加三个函数

​	无参构造函数，拷贝构造函数，析构函数

如果我们写了有参构造，编译器就不会提供无参构造，但是还会提供拷贝构造

如果我们写了拷贝构造函数，编译器就不会给我们提供其他构造函数。





### 深浅拷贝

```c++
// 系统默认的拷贝构造函数实现为浅拷贝，如果如下使用会出问题；
// 解决方法，可以自己定义拷贝构造函数来实现深拷贝
class Person{

    public:

    Person(int n){
        num = new int(n);	// 在堆区开辟内存
    }

    ~Person(){
        if(num != NULL){
            delete num;
            num == NULL;
        }
    }

    int age;
    int *num;
};

int main(){

    Person p1(10);
    Person p2(p1);

    // system("pause");
    return 0;
}



// 解决方案： 加上如下构造函数
    Person(const Person &p){
        age = p.age;
        num = new int(*p.num);
    }
```





### 初始化列表

```c++
class Person{

    public:
    // 初始化列表，来初始化属性
    Person(): age(10), num(10) {}

    // 另外一种更灵活的写法
    Person(int a, int b): age(a), num(b){}


    int age;
    int num;
};

int main(){

    Person p1;
    cout << p1.age << endl;
    cout << p1.num << endl;

    Person p2(10, 100);

    // system("pause");
    return 0;
}

```

```c++
// 对象属性套对象
class Phone{
    public:
    Phone(string s){
        name = s;
    }
    string name;
};


class Person{

    public:
    // phone(s) 相当于是 Phone phone = s; 隐式构造对象
    Person(int a, string s): age(a), phone(s){}


    int age;
    Phone phone;
};

int main(){

    Person p1(10, "hh");
    cout << p1.age << endl;
    cout << p1.phone.name << endl;


    // system("pause");
    return 0;
}
```



### 静态成员

```c++
class Person{

    public:

    static void print(){
        cout << "hello world" << endl;
        age = 100;
    }

    static int age;
};
int Person::age = 10;	// 静态成员变量需要在类外进行初始化

int main(){

    // 1
    Person p;
    p.print();

    // 2
    Person::print();

    cout << Person::age << endl;

    // system("pause");
    return 0;
}
```





### 对象模型

```c++
class Person{}
// c++ 编译器会为每个空对象分配一个字节的内存空间，是为了区分不同的空对象

class Person{int age;}
// 占用四个字节的空间
// 静态成员变量和非静态成员方法不属于类对象上。
```





### this指针

```c++
// this的第一个用途
class Person{
    
    public:
    Person(int age){
        this->age = age;
    }
    int age;
};


// this的第二个用途
class Person{

    public:

    // 返回 Person 引用
    Person& add(int num){
        this->num += num;
        return *this;
    }

    int num = 10;
};

int main(){

    Person p;
    p.add(10).add(10).add(10);

    cout << p.num << endl;

    // system("pause");
    return 0;
}
```





### 空指针与成员函数

```c++
class Person{

    public:

    void showNum1(){
        cout << "show num1" << endl;
    }
    void showNum2(){
        
        // 可以加条件判断，下面就不会报错
        if(this == NULL){
            return;
        }
        
        cout << "show num2" << num << endl;
    }

    int num = 10;
};

int main(){

    Person *p = NULL;
    p->showNum1();  // 正常执行
    p->showNum2();  // 会报错

    // system("pause");
    return 0;
}
```





### 常函数与常对象

​	成员函数加 const 修饰之后，成员函数内部不可以修改成员属性，

​	成员属性加 mutable 关键字后，在常函数中就可以修改了

​	声明对象的时候加const关键字，常对象只能调用常函数

```c++
class Person{

    public:

    void test() const{
        num = 100;
    }

    mutable int num = 10;
};

int main(){

    const Person p;
    p.test();
    p.num = 1000;

    // system("pause");
    return 0;
}
```





### 友元

全局函数做友元	（全局函数可以访问类中的私有成员）

类做友元	（一个类中可以访问另一个类中的私有成员）

成员函数做友元

```c++
class Person{

    // 全局函数做友元，可以访问私有成员
    friend void print(Person *p);
private:
    void test() const{
        cout << " hello world " << endl;
    }
};

void print(Person *p){
    p->test();
}

int main(){

    Person p;
    print(&p);

    // system("pause");
    return 0;
}
```



```c++
class Student{

    // Person 类可以访问本类的私有成员
    friend class Person;
public:
    Student(){}
private:
    void print(){
        cout << " student private " << endl;
    }
};



class Person{
public:
    // 声明，定义写在类外
    Person();
    
    Student *s;

    void test(){
        cout << " hello world " << endl;
        s->print();
    }
};

// 在类外写成员函数
Person::Person(){
    s = new Student;
}


int main(){

    Person p;
    p.test();

    // system("pause");
    return 0;
}
```

```c++
friend class Person::test();
只允许Person类中的test访问 student类中的私有属性
```





### 运算符重载

加号运算符重载

```c++
class Person{
public:
    int a;
    int b;

    // 成员函数
    Person operator+(Person p){
        Person res;
        res.a = this->a + p.a;
        res.b = this->b + p.b;

        return res;
    }
};

// 全局函数
// Person operator+(Person p1, Person p2){
//     Person res;
//     res.a = p1.a + p2.a;
//     res.b = p1.b + p2.b;
//     return res;
// }


int main(){

    Person p;
    p.a = 10;
    p.b = 10;
    Person res = p + p;


    // system("pause");
    return 0;
}

```



<< 运算符重载

```c++
class Person{
public:
    int a;
    int b;
};

// << 运算符都使用全局函数重载, operator<<(cout, p) -> cout << p
ostream& operator<<(ostream &cout, Person &p){
    cout << p.a << " " << p.b;
    return cout;
}

int main(){

    Person p;
    p.a = 10;
    p.b = 10;

    cout << p << endl;


    // system("pause");
    return 0;
}
```



++ 递增运算符

```c++
class MyInteger{
    public:
    int num;

    // 前置加加 ++a
    MyInteger& operator++(){
        this->num++;
        return *this;
    }

    // 后置加加 a++ , 编译器靠占位参数来区分前置后置
    MyInteger operator++(int){
        MyInteger tmp = *this;
        num ++;
        return tmp;
    }
};
```



赋值运算符重载

```c++
class Person{
    public:
    Person(int n){
        num = new int(n);
    }
    int *num;

    // 手动进行深拷贝
    // 如果想要实现 a=b=c 这种效果，需要将 *this返回
    void operator=(Person &p){
        if(this->num != NULL){
            delete num;
            num = NULL;
        } 
        num = new int(*p.num);
    }

    ~Person(){
        if(this->num != NULL){
            delete num;
        }
    }
};

int main(){

    Person p1(10);
    Person p2(100);
    p2 = p1;
    cout << *p2.num << endl;
    // system("pause");
    return 0;
}
```



关系运算符重载

```
bool operator==(..){...}
```



函数调用运算符重载

```c++
// () 即函数调用运算符， 又称仿函数，参数类型以及返回值类型都随意

class Person{
    public:
    void operator()(string s){
        cout << s << endl;
    }
};

int main(){

    Person p;
    p("hello world");

    // system("pause");
    return 0;
}
```



### 匿名对象

```
Person p;	// 有名对象
Person()	// 匿名对象
```





### 继承

语法： class 子类: 继承方式 父类{}

继承方式： 公共继承，保护继承，私有继承

![image-20200802181715846](../../AppData/Roaming/Typora/typora-user-images/image-20200802181715846.png)



继承中的对象模型

​	父类中所有的非静态成员都会继承到子类



继承中构造和析构顺序

​	先构造父亲在构造儿子，先析构儿子，在析构父亲



同名成员处理

```c++
class Person{
    public:
    int num = 10;
};

class Student: public Person{
    public:
    int num = 100;
};

int main(){

    Student s;
    cout << s.num << endl;
    // 访问父类中的num
    cout << s.Person::num << endl;

    // system("pause");
    return 0;
}

/*
	如果子类中出现和父类同名的成员函数，子类中的同名成员函数会隐藏掉父类中所有的同名成员函数
	如果向访问父类中被隐藏的函数需要加作用域
	s.Person::func()
 */
```



同名静态成员

```c++
class Person{
    public:
    static int num;
};
int Person::num = 10;

class Student: public Person{
    public:
    static int num;
};
int Student::num = 100;

int main(){

    Student s;
    cout << s.num << endl;
    cout << s.Person::num << endl;

    cout << Student::num << endl;
    cout << Student::Person::num << endl;

    // system("pause");
    return 0;
}
```



多继承

class A: public B, public C{}

当父类中出现同名成员，需要指明作用域；



菱形继承

```c++
class A{
    public:
    int num;
};

// 继承前加 virtual 关键字后变成了虚继承
// 此时 A 也成为虚基类
class B: virtual public A{};
class C: virtual public A{};

// 加了virtual关键字后，num在D中就只会出现一份
class D: public B, public C{
};




int main(){

    D d;
    d.num = 10;
    // d.num   // 报错，需要加作用域
    cout << d.num << endl;


    // system("pause");
    return 0;
}

```



### 多态

```c++
class Animal{
    public:
    virtual void speak(){
        cout << "animal :" << endl;
    }
};

class Cat: public Animal{
    public:
    void speak(){
        cout << "cat : " << endl;
    }
};

// 地址早绑定，编译阶段就已经确定好函数地址
// 如果需要晚绑定，需要在speak函数前面加 virtual
void test(Animal &animal){
    animal.speak();
}

int main(){

    Cat cat;
    test(cat);
    return 0;
}

/*
 	 动态多态的满足条件
 	 1. 有继承关系
 	 2. 子类重写父类的虚函数
 */
```

```
原理：
	当Animal中speak函数用 virtual修饰之后，对象模型中就会出现一个虚函数指针vfptr指向虚函数表 vftable，
	子类继承Animal之后并且重写虚函数之后会继承虚函数指针重写虚函数表。
```



**纯虚函数**

virtual 返回值类型 函数名(参数列表) = 0;

当有了一个纯虚函数，这个类就是抽象类，不能被实例化，子类必须重写纯虚函数，否则也属于抽象类





### 虚析构和纯虚析构

多态使用时，如果子类中有属性开辟到堆区，那么父类指针在释放时无法调用到子类的析构函数

解决方式：将父类中的析构函数改为虚析构或者纯虚析构

区别：

​	如果类中有了纯虚析构，类就属于抽象类，不能被实例化

```c++
class Animal{
    public:
    // 纯虚函数
    virtual void speak() = 0;

    // 虚析构函数
    // virtual ~Animal(){
    //     cout << "Animal 析构函数 ~" << endl;
    // }

    // 纯虚析构函数, 还需要在外面对析构函数进行定义以清理Animal中的垃圾
    virtual ~Animal() = 0;

};

Animal::~Animal(){
    cout << "Animal 纯虚析构函数 " << endl;
}


class Cat: public Animal{
    public:

    Cat(string name){
        this->name = new string(name);
    }

    void speak(){
        cout << "cat : " << endl;
    }

    string *name;

    ~Cat(){

        cout << " cat 析构函数 ~" << endl;
        if(this->name != NULL){
            delete name;
            name = NULL;
        }
    }
};

void test(Animal &animal){
    animal.speak();
}

int main(){

    Cat cat("xiaoxuan");
    test(cat);
    return 0;
}

```





### 文件操作

ifstream 读文件

ofstream	写文件

fstream	读写文件

```c++
/*
	文件打开方式:
		ios::in		为读文件而打开文件
		ios::out	
		ios::ate	初始位置，文件尾
		ios::app	追加方式
		ios::trunc	如果文件存在先删除在创建
		ios::binary 二进制方式
		
	打开文件方式 可以使用 | 来连接
 */
 
 #include<fstream>


int main(){

    // 写文件
    // ofstream ofs;
    // ofs.open("test.txt", ios::out);

    // ofs << "hello world" << endl;
    // ofs << "hello world" << endl;

    // ofs.close();


    // 读文件
    ifstream ifs;
    ifs.open("test.txt", ios::in);

    if(!ifs.is_open()){
        cout << "error" << endl;
        return 0;
    }

    // 第一种读取方式
    // char buf[1024] = {0};
    // while(ifs >> buf){  // 读到末尾的时候会返回假
    //     cout << buf << endl;
    // }

    // 第二种方式
    // char buf[1024] = {0};
    // while (ifs.getline(buf, sizeof(buf)))
    // {
    //     cout << buf << endl;
    // }
    
    // 第三种方式
    // string buf;
    // while (getline(ifs, buf))
    // {
    //     cout << buf << endl;
    // }

    // 第四种方式
    char c;
    while ((c = ifs.get()) != EOF)
    {
        cout << c << endl;
    }
    
    


    ifs.close();

    return 0;
}
```



```c++
class Person{
    public:
    int age;
};


int main(){

    // 写文件
    // ofstream ofs;
    // ofs.open("test.txt", ios::out | ios::binary | ios::trunc);

    // Person p;
    // p.age = 10;
     
    // ofs.write((const char *)&p, sizeof(p));

    // ofs.close();


    // 读文件
    ifstream ifs("test.txt", ios::in | ios::binary);

    if(ifs.is_open()){

        Person p;
        ifs.read((char *) &p, sizeof(Person));

        cout << p.age << endl;

        ifs.close();
    }

    return 0;
}
```





### 模板

c++另一种编程思想叫泛型编程，主要利用模板， c++提供两种模板机制 函数模板类模板

```c++
// 函数模板
// 声明一个模板，告诉编译器 T 是一个通用的数据类型
// typename 可以换成 class
template<typename T>    
void mySwap(T &a, T &b){
    T tmp = a;
    a = b;
    b = tmp;
}


int main(){

    int a = 10;
    int b = 20;
    // 两种方式使用函数模板
    // 1. 自动类型推导
    mySwap(a, b);

    // 2. 显示指定类型
    mySwap<int>(a, b);

    return 0;
}
```



普通函数与模板函数的区别：

```properties
普通函数在调用的时候可以发生自动类型转换
函数模板在调用时，如果利用自动类型推导，不会发生隐式转换
如果利用显式指定类型的方式，可以发生隐式转换
```



普通函数与函数模板的调用规则：

```c++
如果函数模板和普通函数都可以实现，优先调用普通函数
可以通过空模板参数列表来强制调用函数模板
函数模板也可以发生重载
如果函数模板可以产生更好的匹配，优先调用函数模板


void myPrint(int a, int b){
    cout << " hello 普通函数" << endl;
}

template<typename T>    
void myPrint(T &a, T &b){
    cout << " hello 函数模板" << endl;
}


int main(){

    int a = 10;
    int b = 20;
    myPrint(a, b);

    // 通过空模板参数列表,强制调用函数模板
    myPrint<>(a, b);

    char c = 'a';
    char d = 'b';
    // 使用函数模板更加方便，不用进行类型转换
    myPrint(c, d);

    system("pause");
    return 0;
}

```



类模板

```c++
template<typename NameType, typename AgeType>
class Person{
    public:
    Person(NameType n, AgeType a){
        this->age = a;
        this->name = n;
    }
    NameType name;
    AgeType age;
};

int main(){

    Person<string, int> p("hello", 19);
    cout << p.name << endl;

    system("pause");
    return 0;
}
```



类模板与函数模板的区别：

```properties
1. 类模板没有自动类型推导的使用方式
2. 类模板在模板参数列表中可以有默认值
	template<typename NameType, typename AgeType = int>
```



类模板成员函数与普通类中成员函数的创建时机：

​		普通类中成员函数一开始就创建

​		类模板中成员函数在调用时才创建



类模板对象做函数参数

```c++
1. 指定传入类型
2. 参数模板化
3. 整个类模板化


template<typename NameType, typename AgeType = int>
class Person{
    public:
    Person(NameType n, AgeType a){
        this->age = a;
        this->name = n;
    }
    NameType name;
    AgeType age;
};

// 1. 指定参数类型
void test1(Person<string, int> &p){
    cout << p.name << endl;
}

// 2. 参数模板化
template<class T1, class T2>
void test2(Person<T1, T2> &p){
    cout << p.name << endl;
}

// 3. 将参数类模板化
template<class T>
void test3(T &p){
    cout << p.name << endl;
}

int main(){

    Person<string, int> p("hello", 19);

    test1(p);
    test2(p);
    test3(p);

    return 0;
}

```



类模板与继承

```c++
template<class T>
class A{};

// 第一种方式，指定父类的参数类型
class B: public A<int>{
};


// 第二种方式，将子类声明为类模板
template<class T1>
class C: public A<T1>{};
```



类模板成员函数类外实现

```c++
template<class T>
class A{
    public:
    void myPrint();
};

template<class T>
void A<T>::myPrint(){
    cout << "hello world !" << endl;
}
```



类模板分文件编写

```c++
类模板成员函数创建时机是在调用阶段，导致分文件编写时链接不到
解决：
	1. 直接包含.cpp源文件
	2. 声明和实现写道同一个文件中，并更改后缀名为.hpp,.hpp是约定的名称，不是强制
	

// 第一种方式 =============================================================================
// --- Person.h
#include<iostream>
using namespace std;

template<class T>
class Person{

    public:
    Person(T name);

    T m_Name;
};

// --- Person.cpp
#include <Person.h>

template<class T>
Person<T>::Person(T name){
    this->m_Name = name;
}

// --- main
#include<iostream>
using namespace std;
#include "./Person.cpp"

int main(){

    Person<string> p("he");
    cout << p.m_Name << endl;
    return 0;
}


// 第二种方式  =============================================================================
// --- Person.hpp
#include<iostream>
using namespace std;

template<class T>
class Person{

    public:
    Person(T name);

    T m_Name;
};

template<class T>
Person<T>::Person(T name){
    this->m_Name = name;
}

// --- main
#include<iostream>
using namespace std;
#include "Person.hpp"

int main(){

    Person<string> p("he");
    cout << p.m_Name << endl;
    return 0;
}
```



全局函数和友元

```c++
全局函数类内实现：直接在类内声明友元即可
全局函数类外实现：需要提前让编译器知道全局函数的存在

template<class T>
class A{

    // 类内写的全局函数
    friend void myPrint(A a){
        cout << a.name << endl;
    }

    public:
    A(T n){
        this->name = n;
    }

    private:
    T name;
};


int main(){

    A<string> a("xiauxan");
    myPrint(a);
    return 0;
}
```





类模板案例

```
实现通用的数组：
	可以对内置数据类型以及自定义数据类型进行存储
	将数组中的数据存储到堆区
	构造函数可以传入数组容量
	提供对应的拷贝构造函数以及operator= 防止浅拷贝问题，（= 赋值运算符）
	提供尾插法和尾删法
```




