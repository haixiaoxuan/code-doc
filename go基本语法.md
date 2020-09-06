##### 文档地址

​	https://studygolang.com/pkgdoc



##### 环境配置

​	GO_ROOT、PATH、GO_PATH



##### 命令行模式

​	go build xx.go	生成可执行文件

​	go run xx.go		编译 + 执行



##### 常用转移字符

```go
	// \r 替换掉本行之前的输入
	fmt.Print("\r=")
	fmt.Print("\r==")
	fmt.Print("\r===")
```



##### printf使用

```
	%v 	代表原始输出
	%T	输出变量类型
	%c	输出字符，而非字符所对应的码值
	%p	地址类型
```



##### 变量声名

```go
	var name, age = "xiaoxuan", 18
	println(name, age)
```



##### 常量定义

```go
	const name = "abc"
	const (
		age = 19
		value = true
	)
	fmt.Println(name, age, value)
```



##### 基本数据类型

```go
整数类型： int,int8,int16,int32,int64,uint,uint8,uint16,uint32,uint64,byte
浮点类型： float32,float64
字符类型： 使用 byte
布尔值:  bool
字符串:  string	go的字符串是由单个byte连接起来的，go统一使用utf8，字符串是不可变类型
				  `hh\nhh`	反引号字符串中的字符会原样输出
特殊： rune 与int32等价，表示一个unicode码
```



##### 值类型

```go
int系列，float系列，bool，string，数组和结构体（内存通常在栈中分配）

引用类型：
	指针、slice切片、map、管道channel、interface等
```



##### 查看变量所占空间大小

```go
	name := '解'
	fmt.Println(unsafe.Sizeof(name))
```



##### 类型转换

```go
	// 数值类型转换
	var age int = 18
	var newAge float32 = float32(age)
	fmt.Printf("%T %v \n", newAge, newAge)

	// 数值 -> string
	// 方式1  Sprintf
	var num1 float32 = 3.14
	var num1Str = fmt.Sprintf("%.2f", num1)
	fmt.Println(num1Str)

	// 方式2  strconv
	var age2 = 18
	var age2Str = strconv.FormatInt(int64(age2), 10)
	fmt.Println(age2Str)

	// str -> 数值类型
	var value, _ = strconv.ParseInt("100", 10, 64)
	fmt.Printf("%T %v", value, value)

```



##### 指针

```go
	&num	获取变量地址
	var p *int		定义指针变量，存放的是指针
	*p		获取指针p指向的值
```



##### 自定义包

```
	import时从 GO_ROOT/src下开始，到最后一个文件夹结束
		import go_code/xx/xx/model
	使用时
		model.xx()
```



##### 交互输入

```go
	var name string
	var age int

	// 使用scanln
	fmt.Scanln(&name)
	fmt.Scanln(&age)
	fmt.Println(name, age)

	// 使用 scanf
	fmt.Scanf("%s %d", &name, &age)
	fmt.Println(name, age)
```



##### 流程控制

```go
	// if
	var flag = 19
	if flag > 18 {
		fmt.Println("hello")
	}

	if flag2 := 20; flag2 > 10 {
		fmt.Println("world")
	}


	// switch	匹配后面不需要加 break
	/*
		1. case 后面可以跟多个表达式	case 表达式1,表达式2 :
		2. default 不是必须的
		3. switch后面也可以不加表达式，类似多个if/else来使用
		4. 如果在 case语句块后面加 fallthrough, 则会执行下一个case
	 */
	switch 1 {
		case 1:
			fmt.Println("1~")
			fallthrough
		default:
			fmt.Println("default!")
	}

	// 6 可以进行类型判断
	var x interface{}
	var y = 10.0
	x = y
	switch i := x.(type) {
		case nil:
		fmt.Println("type = nil ", i)
		case float32:
		fmt.Println("type = float32 ", i)
		case float64:
		fmt.Println("type = float64 ", i)
	}

```



##### 循环

for

```go
	for i := 0; i < 10; i++ {
		fmt.Println(i)
	}

	// 下面两种写法等价
	for{
		break
	}
	for ;; {
		break
	}


	// for-range方式
	// 如果采用这种方式对中文进行遍历，会乱码，需要将str转为 []rune 切片
	str := "hello world"
	for i := 0; i < len(str); i++ {
		fmt.Printf("%c \n", str[i])
	}

	// 按照字符方式来遍历，非字节（index 是字节索引）
	str1 := "hello 中国"
	for index, val := range str1 {
		fmt.Printf("%d  %c \n", index, val)
	}
```

while

```go
	// go中没有实现 while，do-while
```

标签的使用

```go
	lab1:
	for ;; {
		break lab1
	}
```



##### 函数

```go
/*
	1. go函数不支持重载
	2. 值类型的参数默认是进行值拷贝
	3. 在go中函数也是一种数据类型，可以赋给变量
 */
func sum(a int, b int) (int, int)  {
	return a + b, a -b
}


// 自定义数据类型的用法
type myInt int
type myFunc func(int, int) (int, int)

func test(f myFunc) (int, int) {
	return f(10, 11)
}

// 支持对函数返回值命名， res1 res2 已经定义，不用在被重新定义
func cal(a int, b int)(res1 int, res2 int) {
	res1 = a + b
	res2 = a - b
	return
}

// 可变参数
func test2(args ... int) int {
	sum := 0
	for i := 0; i < len(args); i++ {
		sum += args[i]
	}
	return sum
}
```

init函数

```go
// 每一个源文件都可以有一个init函数，在main函数执行之前被go调用执行
func init(){
	fmt.Println("begin ... ")
}
```

匿名函数

```go
	var f = func(a int, b int) int {
		return a + b
	}
	fmt.Println(f(10, 11))
```



##### 闭包

```go
// 闭包就是函数与其相关的引用环境组合的一个整体
func addUpper(a int) func(int) int{
	var n = 10
	return func(x int) int {
		n = n + x
		return n
	}
}
/*
	返回的是一个匿名函数，但是这个匿名函数用到了函数外部的n，因此匿名函数和n形成一个整体，构成闭包
 */

func main() {

	f := addUpper(10)
	fmt.Println(f(1))	// 11
	fmt.Println(f(2))	// 13
	fmt.Println(f(3))	// 16
}
```



##### defer

```go
// 延时机制，在程序执行完成后释放资源
func test(){
    
    // 当执行到defer的时候暂不执行，压入单独的栈defer栈 （相关的值变量也会同时拷贝入栈）
    // 函数执行完成之后再按照先入后出的方式执行 defer栈
	defer fmt.Println("defer")
	fmt.Println("test")
}

```



##### 字符串函数

```go
	/*
		1. 按字节统计字符串长度 len(str)
		2. 字符串遍历，同时处理有中文问题 r:= []rune(str), 然后用 for 循环处理
		3. 字符串转整数	value, error := strconv.Atoi("10")
		4. 整数转字符串	str := strconv.Itoa(10)
		5. 字符串转 []byte	bytes := []byte("hello world")
		6. []byte 转字符串	str := string([]byte{123,124,125})
		7. 10进制转2,8,16	strconv.FormatInt(123, 2)
		8. 包含关系		strings.Contains("abc", "a")
		9. 子串出现次数	strings.Count("abc", "a")
		10. 不区分大小写的字符串比较	strings.EqualFold("a", "A"), 区分大小写使用 ==
		11. 返回子串在字符串中第一次出现的index	strings.Index("abc", "b"), 没有返回 -1
		12.										strings.LastIndex(x,x)
		13. 将子串替换为另外一个子串	strings.Replace("", "", "", -1)		-1 表示全部替换
		14. strings.Split(x, x)
		15. strings.ToLower()
		16. strings.TrimSpace()
		17. strings.Trim(x, x)		去除掉字符串两边的指定字符
		18. strings.TrimLeft 	strings.TrimRight
	 */
```



##### 日期函数

```go
	currentTime := time.Now()
	fmt.Printf("%T %v \n", currentTime, currentTime) 	// time.Time  2020-06-17 09:06:53.1159617

	fmt.Println("年：", currentTime.Year())
	fmt.Println("月：", currentTime.Month(), int(currentTime.Month()))
	fmt.Println("日：", currentTime.Day())
	fmt.Println("时：", currentTime.Hour())
	fmt.Println("分：", currentTime.Minute())
	fmt.Println("秒：", currentTime.Second())


	// 格式化
	// 1. 依次把年月日时分秒拿出来格式化
	// 2. 第二种
	fmt.Printf(currentTime.Format("2006/01/02 15:04:05"))
	// 2006 01 02 15 04 05 这些数字是固定组合


	// 时间常量 ... 可以用户获取指定时间单位的时间
	second := time.Second
	minute := time.Minute
	fmt.Println(second, minute)


	// 休眠
	time.Sleep(1 * time.Second)


	// 获取时间戳
	timestamp := time.Now().Unix()
	timestampNano := time.Now().UnixNano()
	fmt.Println(timestamp, timestampNano)

```



##### 内置函数

```go
内置函数统一在文档中  builtin 中
	len		求长度
	new 	分配内存，主要用来分配值类型
	make 	分配内存，主要用来分配引用类型

```



##### 错误处理

普通异常处理

```go
func test(){

	/*
		go 不支持 try catch
		go 引入 defer panic recover
		go 中可以抛出一个 panic 异常，然后再 defer 中通过recover捕获这个异常，然后正常处理
	 */
	defer func() {
		error := recover()
		if error != nil {
			fmt.Println("error")
		}
	}()
	num1 := 10
	num2 := 0
	res := num1 / num2
	fmt.Println("res: ", res)
}
```

自定义异常

```go
func test(name string) (error) {
	if name == "abc" {
		return nil
	}else{
		return errors.New("name is error")
	}
}


func main() {

	/*
		自定义错误
		errors.New("错误信息")，返回一个error类型的值，表示一个错误
		panic内置函数，可以接受error类型的变量，输出错误信息，并退出程序

	 */

	error := test("ab")
	if error != nil {
		// 输出错误信息，并终止程序
		panic(error)
	}

	fmt.Println("over")	// 此处不会被执行
}

```



##### 数组

```go
	// 数组，值类型的数据
	var arr1 [5]int = [5]int{1, 2, 3, 4, 5}
	var arr2 = [5]int{1, 2, 3, 4, 5}
	var arr3 = [...]int{1, 2, 3, 4, 5}
	var arr4 = [3]string{1:"a", 0:"b", 2:"c"}

	fmt.Println(arr1)
	fmt.Println(arr2)
	fmt.Println(arr3)
	fmt.Println(arr4)


	// 初始化之后赋值
	var arr [2]int
	arr[0] = 1
	arr[1] = 2


	// 遍历
	// 1. 普通for循环遍历 	2. for-range 遍历
	for index, value := range arr {
		fmt.Println(index, value)
	}


	/*
		注意事项
		1. 长度固定不能动态变化
		2. 数组中的元素可以是值类型或者引用类型
		3. 如果在其他函数中想修改此数组，需要使用引用传递的方式
		4. 长度是数组类型的一部分，传递参数时要考虑。不能把[2]int 传递给 [3]int
	 */
```



##### 切片

```go
	// 切片是引用类型，是数组的一个引用
	var intArr = [...]int{1, 2, 3, 4, 5}

	// 声明一个切片, 此处为左闭右开
	sli := intArr[1:3]
	fmt.Println(sli)
	fmt.Println(len(sli))
	fmt.Println(cap(sli))	// 切片的容量是可以动态变化的


	// 切片的使用
	var slice []int = make([]int, 4, 10)	// 长度为4，容量为10（容量为可选参数）
	fmt.Println(slice)
	fmt.Printf("长度：%d  容量：%d \n", len(slice), cap(slice))
	slice[0] = 10
	slice[1] = 100
	fmt.Println(slice)


	// 直接初始化
	var slice2 []int = []int{1, 2, 3}
	fmt.Println(slice2)


	/*
		切片注意事项：
		1. 对数组截取时可以简写，[0:3] -> [:3], 到末尾也同样可以简写[3:], [:]
		2. 切片定义完成之后还不能使用，需要引用一个数组或者make一个空间来供切片使用
		3. 切片可以继续切片

	 */


	// 对切片进行 append 操作 (底层操作即为创建新的数组，将旧的数据拷贝)
	slice2 = append(slice2, 4, 5)		// 追加具体元素
	var slice3 []int = []int{6, 7}
	slice2 = append(slice2, slice3...)		// 追加切片
	fmt.Println(slice2)


	// 切片的拷贝操作
	var slice4 = make([]int, 10)
	copy(slice4, slice2)
	fmt.Println(slice4)


	/*
		string 底层是 byte 数组，因此string也可以进行切片处理
		string 是不可变的，不能通过 s[0] = 'z'来修改
		如果要修改可以先转成 []byte, 修改之后再赋值
	 */
	arr := []byte("abc")
	arr[0] = 'e'
	s := string(arr)
	fmt.Println(s)

	// rune 是按照字符来进行处理，可以处理中文
	arr2 := []rune("abcd")
	arr2[0] = '中'
	s2 := string(arr2)
	fmt.Println(s2) 	// 中bcd
```



##### map

```go
	// map 声明, 声明不会分配内存，分配内存之后才能赋值和使用, size 为可选
	var m map[string]string
	m = make(map[string]string, 10)
	m["n1"] = "x1"
	m["n2"] = "x2"

	// 声明时直接make
	var m2 map[string]string = make(map[string]string)

	// 声明时直接赋值
	var m3 map[string]string = map[string]string{"1": "a", "2": "b", }
	fmt.Println(m, m2, m3)


	// 增加, 如果key存在就是修改，如果key不存在就是添加
	m3["1"] = "c"

	// 删除, 如果key存在就是删除，如果不存在就什么也不做
	delete(m3, "1")

	// 清空, 只能遍历map，将key逐个删除，或者将引用赋新值，原来的值就会被gc回收

	// 查找
	val, findRes := m3["2"]
	if findRes {
		fmt.Println("key=1, value=", val)
	}else{
		fmt.Println("没有找到！")
	}


	// 遍历
	for k, v := range m3 {
		fmt.Println(k, v)
	}


	// map 切片
	var mapSlice []map[string]string
	mapSlice = make([]map[string]string, 2)
	if mapSlice[0] == nil {
		mapSlice[0] = make(map[string]string, 2)
		mapSlice[0]["name"] = "tom"
	}
	fmt.Println(mapSlice)
	// 如果要设置mapSlice[2], 则需要使用 append来增加, 否则会越界


	// 如果想将map进行排序输出，则需要先将key拿出来排序，再按照key来进行遍历
	// 此处的可以不需要初始化，append的时候会自动进行make
	var keys []string
	for k, _ := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Println(keys)

	/*
		map注意事项：
		1. map会自动扩容，不会抛异常
	 */
```



##### 结构体

```go
type student struct {
	name string
	age int
}

type person struct {
	Name string `json:"name"`
	Age int `json:"age"`
}

func main() {

	/*
		go中没有class，通过struct来完成oop
		1. 如果没有对变量进行初始化，则都会有默认值
	 */

	// 声明
	var stu student
	// 声明的第二种方式
	var stu2 student = student{"abc", 18}
	// 方式三
	var stu3 *student = new (student)
	// 方式四
	var stu4 *student = &student{"bcd", 19}
	fmt.Println(stu, stu2, *stu3, stu4)

	/*
		方式3，4 返回的是结构体指针，结构体指针访问字段的标准方式是：(*stu3).name
		但是go做了简化，也支持 stu3.name, go编译器对这种方式进行了转换
		note：
			不能这样写 *stu3.name
			因为 . 的优先级比 * 高

		结构体的注意事项：
		1. 结构体的所有字段在内存中是连续的
		2. 结构体是用户单独定义的数据类型，和其他类型进行转换时，需要保证完全相同的字段（个数类型名称）
		3. struct的每个字段上，可以写一个tag，该tag可以通过反射机制获取，常见的场景就是 序列化和反序列化

	 */

	// 格式化为json串
	var per person = person{"效玄", 18}
	data, err := json.Marshal(per)
	if err == nil {
		fmt.Println(string(data))
	}
}
```

方法

```go
type student struct {
	name string
	age int
}

func (stu student) print(){
	fmt.Println(stu.name, stu.age)
}

func (stu *student) String() (string) {
	return (*stu).name
}

func main() {

	/*
		go 中的方法是作用在指定数据类型上，不仅仅是 struct 可以有方法
		struct为值类型的参数，通过值拷贝完成传递
	 */
	var stu = student{"abc", 18}
	stu.print()
	fmt.Println(&stu)

	/*
		注意细节：
		如果一个变量实现了 String 方法，则 fmt.Print 会调用
		(&stu).String 等价于 stu.String , 同样是编译器做的优化
	 */

	/*
		方法和函数的区别：
		调用方式不同： 函数名(实参)	变量.方法名(实参)
		对于普通函数，参数为值类型时，不能将引用直接传递。 方法不一样
	 */

}
```



##### 工厂模式

```go
type student struct {
	name string
	age int
}

// 如果 student 为小写，则可以使用这种工厂模式
func GetStudent(name string, age int) *student {
	return &student{name, age}
}
```



##### 继承

```go
type Person struct{
	name string
}

type Student struct {
	Person		// 匿名结构体		p person 这种格式为有名结构体
	age int
}

func main() {

	// 继承
	// 如果一个struct中嵌套了另外一个匿名结构体，从而实现继承特征，可以使用匿名结构体的所有字段和方法
	var stu = Student{}
	stu.age = 18
	stu.Person.name = "abc"
	fmt.Println(stu)

	/**
		note:
		当与匿名结构体有相同的字段和方法时采取就近原则
		如果两个匿名结构体拥有相同的字段或方法，访问时必须指定匿名结构体的名称，不然编译报错
		如果一个struct嵌套了一个有名结构体，这种模式成为组合，那么在访问有名结构体的字段或方法时，必须加上有名的名称
	 */
	
	// 另外一种初始化方式
	var stu2 = Student{ Person{"abc"}, 18}
	fmt.Println(stu2)
}
```



##### 接口

```go
type Usb interface {
	start()
}

type Phone struct {

}

func (phone Phone) start(){
	fmt.Println("phone usb start!")
}

func main() {

	/*
		interface 可以定义一组方法，并且不需要实现。
		interface 不能包含任何变量
		go 中的接口不需要显式实现，只要一个变量包含接口中的所有方法，那么就认为实现了此接口
	 */

	var p Usb = Phone{}
	p.start()


	/*
		note
		接口本身不能创建实例
		接口中所有的方法都没有方法体
		只要是自定义数据类型都可以实现接口，不仅仅是结构体
		一个自定义类型可以实现多个接口
		一个接口(A)可以继承别的接口，比如说(B,C), 此时如果要实现A，则需要把BC相应的方法也实现了
		interface默认是一个指针(引用类型), 如果没有对interface初始化就会输出 nil
		空接口 interface{} 没有任何方法，所有类型都实现了空接口。 例：var a interface{} = 10
	 */

}
```



##### sort

使用sort对自定义数据类型的切片进行排序

```go
type Student struct {
	Name string
	age int
}

type StudentSlice []Student
func (s StudentSlice) Len() int {
	return len(s)
}
// 按照年龄从小到大排序
func (s StudentSlice) Less(i, j int) bool {
	return s[i].age < s[j].age
}
func (s StudentSlice) Swap(i,j int){
	//tmp := s[i]
	//s[i] = s[j]
	//s[j] = tmp

	s[i],s[j] = s[j],s[i]
}


func main() {

	/*
		需求： 对Student切片进行排序
	 */

	var arr StudentSlice = make(StudentSlice, 3)
	arr[0] = Student{"abc", 18}
	arr[1] = Student{"abc2", 17}
	arr[2] = Student{"abc3", 19}

	sort.Sort(arr)
	fmt.Println(arr)
}
```



##### 类型断言

```go
type Student struct {
	Name string
	Age int
}

func main() {
	
	/*
		switch 中也可以进行类型断言
	 */

	var s interface{}
	var stu1 = Student{"abc", 18}

	s = stu1
	var stu2 Student

	// 这就是类型断言，表示s是否指向Student类型的变量，如果是就执行，否则报错
	stu2 = s.(Student)

	stu3, ok := s.(Student)
	if ok {
		fmt.Println(stu3)
	}

	fmt.Println(stu2)

}
```



##### 文件操作

读

```go
	const (
		bufferSize = 4096
		filePath = "C:\\xiaoxuan\\workspace\\go_workspace\\src\\go-project-test\\src\\main\\utils\\util.go"
	)

	// 打开一个文件并进行操作
	file, error := os.Open(filePath)
	if error != nil {
		return
	}

	// 创建一个reader，带缓冲区的, 默认bufferSize 为 4096，看源码可知
	reader := bufio.NewReader(file)

	for {
		str, err := reader.ReadString('\n')	// 以\n分割
		if err == io.EOF {	// io.EOF 表示文件末尾
			break
		}
		fmt.Println(str)
	}

	file.Close()



	// 一次读取全部内容, 适用于小文件
	// 没有显式open文件，也不用显式close，都被封装在 ReadFile中了
	content, err2 := ioutil.ReadFile(filePath)
	if err2 == nil {
		fmt.Println(string(content))
	}

```

写

```go
	const (
		bufferSize = 4096
		filePath = "C:\\xiaoxuan\\test.log"
	)

	/*
		O_RDONLY int = syscall.O_RDONLY // open the file read-only.
		O_WRONLY int = syscall.O_WRONLY // open the file write-only.
		O_RDWR   int = syscall.O_RDWR   // open the file read-write.
		O_APPEND int = syscall.O_APPEND // append data to the file when writing.
		O_CREATE int = syscall.O_CREAT  // create a new file if none exists.
		O_EXCL   int = syscall.O_EXCL   // used with O_CREATE, file must not exist
		O_SYNC   int = syscall.O_SYNC   // open for synchronous I/O.
		O_TRUNC  int = syscall.O_TRUNC  // if possible, truncate file when opened.
	*/


	// OpenFile 是一个更一般性的函数，第二个参数可以指定打开文件的模式：只读、只写等
	// 第三个参数为权限控制
	file, error := os.OpenFile(filePath, os.O_APPEND | os.O_CREATE, 777)
	defer file.Close()

	if error != nil {
		fmt.Println("error ", error)
		return
	}

	str := "hello world"
	// 使用带缓存的Writer
	writer := bufio.NewWriter(file)
	writer.WriteString(str)

	// 因为write是带缓存的，结束时需要flush
	writer.Flush()
```

判断文件是否存在

```go

func PathExist(filePath string) (bool, error) {

	/*
		判断文件是否存在，使用os.Stat() 返回的错误值来进行判断
			如果错误为nil，说明文件或者文件夹存在
			如果返回的err使用 os.IsNotExist() 判断为true，说明文件夹不存在
			如果返回其他类型，则不确定存不存在
	 */
	
	_, err := os.Stat(filePath)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, nil
}
```

文件拷贝

```go
func copyFile(dsFileName string, srcFileName string) (written int64, err error) {

	srcFile, error := os.Open(srcFileName)
	if error != nil {
		fmt.Println(error)
		return
	}
	defer srcFile.Close()
	reader := bufio.NewReader(srcFile)

	dstFile, error := os.OpenFile(dsFileName, os.O_WRONLY | os.O_CREATE, 0666)
	if error != nil {
		fmt.Println(error)
		return
	}
	defer dstFile.Close()
	writer := bufio.NewWriter(dstFile)

	return io.Copy(writer, reader)
}
```



##### 命令行参数

```go
	// os.Args 是数组切片，存储全部参数
	args := os.Args
	for index, value := range args {
		fmt.Println(index, value)
	}

	
	// -u root -p 1234
	var userName string
	var password string

	// 封装版参数解析
	flag.StringVar(&userName, "u", "", "用户名，默认为空")
	flag.StringVar(&password, "p", "", "密码，默认为空")

	flag.Parse()

	fmt.Println(userName, password)
```



##### json

```go
	// 序列化使用此方法
	data, err := json.Marshal(per)
	
	// 反序列化
	var stu Student
	err := json.Unmarshal([]byte("{\"Name\":\"abc\"}"), &stu)
	if err == nil {
		fmt.Println(stu)
	}
	// 如果反序列化 map等，不需要手动make
```



##### 单元测试

```go
import (
	"testing"
	"go-project-test/src/main/utils"
)


func TestAdd(t *testing.T){
	res := utils.Add(1, 3)

	if res != 4 {
		t.Fatalf("xxxx error")
	}

	t.Logf("successful")
}

	/*
		go 语言自带轻量级的测试框架 testing和自带的test命令来实现单元测试和性能测试
		note:
			测试用例的文件名必须以 _test.go 结尾
			测试用例函数必须以 Test 开头，一般来说就是 Test+被测函数名
			函数的形参类型必须是 *testing.T
			一个测试用例文件中可以有很多个测试函数
			运行命令：
				> go test [运行正确无日志，错误时会有日志]
				> go test -v [都会输出日志]
			当出现错误时，可以使用t.Fatalf来格式化输出错误信息，并退出程序
			t.Logf方法可以输出日志
			测试用例函数没有放在main函数中，但是也执行了，这就是方便之处
			PASS表示测试用例成功，Fail表示测试用例失败
			测试单个文件需要带上被测试的源文件 go test xx_test.go
			测试单个方法 go test xx_test.go TestXx

	 */
```





##### 压力测试

```go
--- main.go
package main

import "fmt"

func test1(n int) int {
	var arr = make([]int, n)
	arr[0], arr[1] = 1, 1
	for i := 2; i < len(arr); i++ {
		arr[i] = arr[i-1] + arr[i-2]
	}
	return arr[n-1]
}

func test2(n int) int {
	if n == 1 {
		return 1
	}
	if n == 2 {
		return 1
	}
	return test2(n-1) + test2(n-2)
}

func main(){
	fmt.Println(test1(50))
}


--- pressure_test.go
package main

import "testing"

/*
	go test -bench=./
 */
func BenchmarkTest1(b *testing.B) {
	b.Log("test1....")
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		test1(30)
	}
}
func BenchmarkTest2(b *testing.B) {
	b.Log("test2....")
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		test2(30)
	}
}

// 运行方式：goland软件，鼠标右击需要进行bench压力测试的文件夹(包)，run -> go bench ...
```







##### 携程

```go
func test() {
	for i := 0; i < 10; i++ {
		fmt.Println("print test .. ")
		time.Sleep(1 * time.Second)
	}
}

func main() {

	/*
		go 携程的特点
			1. 有独立的栈空间
			2. 共相程序堆空间
			3. 调度由用户控制，不需要操作系统干预，也就不需要切换到内核态
			4. 携程是轻量级线程
		note:
			如果主线程执行完成，携程即使没有执行完成也会退出
	 */

	go test()

	for i := 0; i < 10; i++ {
		fmt.Println("print main .. ")
		time.Sleep(1 * time.Second)
	}


	// 获取cpu核数
	cpuNum := runtime.NumCPU()
	fmt.Println("cpu 核数", cpuNum)

	// 设置可以使用的最大 核数，go1.8 之后默认是多核，1.8之前需要设置
	runtime.GOMAXPROCS(cpuNum)

}
```



携程调度

```go
	// 让出携程资源，让其他携程优先执行, 等同于 yield
	// runtime.Gosched()

	//for i := 1; i < 3; i++ {
	//	go func() {
	//		for j := 1; j < 5; j++ {
	//
	//			// 此处这样写的话会有闭包问题，应该将i当作参数传进来
	//			fmt.Printf("携程%d: %d \n", i, j)
	//		}
	//	}()
	//}

	for i := 1; i < 3; i++ {
		go func(num int) {

			if num == 1  {
				runtime.Gosched()
			}

			for j := 1; j < 5; j++ {
				fmt.Printf("携程%d: %d \n", num, j)
			}
		}(i)
	}
```



```
// 携程自杀
// 如果主携程以这种方式自杀，程序不会结束，子携程执行完成之后会报死锁的错
runtime.Goexit() // 退出当前携程
// 退出前会触发 defer
```



##### 读写锁

...



##### 互斥锁

```go
// 运行时增加 -race 参数，可以知道是否会有资源争抢问题
var (
	lock sync.Mutex	
	sum = 0
)

func test(sum *int) {
	lock.Lock()
	for i := 0; i < 10; i++ {
		*sum ++
	}
	lock.Unlock()
}


func main() {

	fmt.Println("begin ... ")

	for num := 0; num < 10; num ++ {
		go test(&sum)
	}

	time.Sleep(1 * time.Second)

	lock.Lock()		// 此处也需要加锁，是因为编译器不知道此处别的携程已经执行完了
	fmt.Println(sum)
	lock.Unlock()
}
```



##### channel

```go
	/*
		channel 本身就是一个数据结构队列，先进先出
		线程安全，多携程访问时不需要加锁
		channel是有类型的，一个string类型的channel只能存放string
		channel是引用类型，必须初始化之后才能写入数据
		写入数据时不能超过channel容量，读取时不能取空channel
		使用close可以关闭channel，关闭之后不能再写入数据，但是仍然可以读取
		channel 支持for range 遍历：
			1. channel如果没有关闭，会出现deadlock error
			2. channel关闭之后会正常遍历数据。
	 */

	var intChannel chan int		// 不只是int，可以是map struct 等
	intChannel = make(chan int, 3)

	fmt.Println(intChannel)

	intChannel<- 1
	fmt.Println(intChannel)

	// 取数据
	fmt.Println(<-intChannel)
```

案例

```go
/*
 	创建两个携程，一个写管道一个读管道，当管道数据读取完成之后发送信号让主线程结束
 */
var (
	numChannel chan int = make(chan int, 10)
	flagChannel chan bool = make(chan bool, 1)
)

func writeData(){
	for i := 0; i < 10; i++ {
		numChannel<- i
		fmt.Println("write data ", i)
	}
	close(numChannel)
}


func readData(){
	for {
		i, ok := <-numChannel
		if !ok {
			break
		}
		time.Sleep(1 * time.Second)
		fmt.Println("read data", i)
	}

	flagChannel<- true
	close(flagChannel)
}


func main() {

	/*
		如果运行时发现一个管道不断地写入数据但是并没有读取，会再写入数据超过cap的时候发生deadlock，
		如果读取速度较慢，则会自动阻塞，不报错
	 */

	go writeData()
	go readData()

	for {
		_, ok := <- flagChannel
		if !ok {
			break
		}
	}
	print("over ! ")

}

```

只读或者只写channel

```go
// 只写channel
func send(channel chan<- int){
	for i := 0; i < 10; i++ {
		channel<- i
	}
	close(channel)
}

// 只读channel
func read(channel <-chan int){
	for {
		i, ok := <-channel
		if !ok {
			break
		}
		fmt.Println(i)
	}
}


func main() {

	/*
		channel 可以声名为只读或者只写,
	 */

	var channel chan int = make(chan int, 10)
	go send(channel)
	go read(channel)

	time.Sleep(time.Second)
	fmt.Println("over!")
}
```

携程中异常处理

```go
func test() {
	defer func() {
		if err := recover(); err != nil {
			fmt.Println(err)
		}
	}()
	var i = 1/0
	fmt.Println(i)
}


func main() {

	// 如果携程中出现panic,但是没有处理,就会引起整个程序崩溃
	go test()

	time.Sleep(time.Second)
	fmt.Println("over")
}
```

select

```go
	var ch1 chan int = make(chan int, 10)
	var ch2 chan string = make(chan string, 5)

	for i := 0; i < 10; i++ {
		if i < 5 {
			ch2 <- "hello" + fmt.Sprintf("%d", i)
		}
		ch1 <- i
	}

	label:
	// 如果不确定管道什么时候关闭,可以使用select,不会进行阻塞
	// 从所有的case中选择一个不阻塞的，如果都阻塞，则走default
	for {
		select {
			case i := <- ch1 :
				fmt.Println(i)
			case i := <- ch2 :
				fmt.Println(i)

			default:
				fmt.Println("都拿不到数据了,可以由自己逻辑")
				break label
            	// 如果只写break，则只会跳出 select
		}
	}

	fmt.Println("over")
```



##### 反射

```go
/*
	reflect.TypeOf
	reflect.ValueOf
 */

func reflectTest(b interface{}){

	// 通过反射获取变量的 type
	rType := reflect.TypeOf(b)
	fmt.Println("rType = ", rType)

	// 获取value
	rValue := reflect.ValueOf(b)
	fmt.Println("rValue = ", rValue)

	// 对value进行运算
	fmt.Println(rValue.Int() + 10)

	// 将value转为 interface{}
	iV := rValue.Interface()
	// 类型断言
	num := iV.(int)
	fmt.Println(num + 20)


	// 通过反射修改变量
	var num2 = 33
	num2Ref := reflect.ValueOf(&num2)
	num2Ref.Elem().SetInt(34)
	fmt.Println("num2 = ", num2)

}


func main() {

	/*
		反射可以在运行时动态获取变量的任何消息，比如类型type，类别kind
		如果是结构体变量，还可以获取到结构体本身的信息(包括结构体的字段\方法)
		通过反射可以修改变量的值，可以调用关联的方法

		常用用法：用静态类型 interface{} 保存一个值，
		通过调用TypeOf获取其动态类型信息，该函数返回一个Type类型值
		调用ValueOf 返回Value类型的值，该值代表运行时的数据。
		Zero接受一个Type类型参数并返回该类型零值的value类型


		note:
			reflect.Value.Kind	获取变量的类别，kind和type可能时相同，也可能不同，比如说type是Student，但是kind是struct
			使用反射方式获取变量的值，要求数据类型匹配，比如x是int，就应该使用ValueOf(x).Int(),否则报panic
	 */

	reflectTest(10)
}
```



```go
import (
	"fmt"
	"reflect"
)

type Person struct{
	Name string
	Age int
}

func main(){

	var p = Person{Name: "xiaoxuan", Age: 19}

	rType := reflect.TypeOf(p)
	fmt.Println(rType.Name())	// Person
	fmt.Println(rType.Kind())	// struct
	fmt.Println(rType.NumField())
	fmt.Println(rType.NumMethod())
	fmt.Println(rType.Field(0)) // field.Name field.Type
	//fmt.Println(rType.Method(0)) // method.Name method.Type
	//fmt.Println(rType.FieldByIndex([]int{0, 1}))	// 找出第0个父结构体中的第一个属性

	pValue := reflect.ValueOf(p)
	fmt.Println(pValue.Field(0))		// 此时拿出来的值不能直接使用，需要调用 .Interface() 方法
	pValue.FieldByName("Name")

    // 指针形式，可以修改原来对象的值
	p2Value := reflect.ValueOf(&p)
	p2Value.Elem()		// 获取地址&p 中的值
	p2Value.Elem().CanSet()	// 检查当前地址value内的值可以改变
	//p2Value.Elem().set
    
    method := p2Value.Elem().MethodByName("test")
    method.call([]reflect.Value{reflect.ValueOf("xx")})	// 调用对象方法
}
```





##### 网络编程

```go
	// TCP
	// 服务端
	listen, err := net.Listen("tcp", "127.0.0.1:5000")
	listen.Accept()

	// 客户端
	conn, err := net.Dial("tcp", "127.0.0.1:5000")


	// UDP
	udp_addr, err := net.ResolveUDPAddr("udp", "127.0.0.1:5001")
	conn, err = net.ListenUDP("udp", udp_addr)

	conn.Close()
	conn.Read()
	conn.Write()
```



```go

	// http
	http.HandleFunc("/hello", func(writer http.ResponseWriter, request *http.Request) {
		writer.Write([]byte("hello world"))
	})

	http.ListenAndServe("127.0.0.1:5000", nil)


	// client
	response, _ := http.Get("http://www.baidu.com")
	fmt.Println(response)

	http.Post("http://www.baidu.com", "application/x-www-form-urlencoded", 
		strings.NewReader("hello_world"))

```





##### 定时器

```go
	// 定时器
	timer := time.NewTimer(3 * time.Second)
	fmt.Println(time.Now())

	// 触发之前可以删除定时器
	//timer.Stop()

	// 定时器重置, 从当前时间开始，2s之后触发
	time.Sleep(2 * time.Second)

	// 此处会阻塞，直到定时器触发
	endTime := <-timer.C
	fmt.Println(endTime)
```

```go
	// 周期定时器
	ticker := time.NewTicker(1 * time.Second)

	for {
		fmt.Println(ticker.C)
	}
```



##### 等待组

可以每开启一个携程加一，每结束一个携程-1，等所有携程都结束的时候主线程结束

```go
	// 等待组
	var waitGroup sync.WaitGroup
	
	// + 1
	waitGroup.Add(1)
	
	// - 1
	waitGroup.Done()
	
	// 等待直到 为 0
	waitGroup.Wait()
```



##### once

```go
func test(){
	fmt.Println("once")
}


func main() {
	
	var once sync.Once

	for i := 0; i < 4; i++ {
		// test 指挥执行一次
		once.Do(test)
	}

	time.Sleep(time.Second)
}
```



##### 信号量

可以通过 固定长度的管道 来实现并发控制



##### condition

```go
func testCond(){

	condition := false
	cond := sync.NewCond(&sync.Mutex{})

	go func() {
		cond.L.Lock()
		condition = true
		cond.Signal()
		//cond.Broadcast() // 通知所有等待的携程
		cond.L.Unlock()
	}()

	cond.L.Lock()
	for !condition {
		cond.Wait()
	}
	cond.L.Unlock()
}

func main() {
	testCond()
}
```





##### 原子操作

只支持基本类型

```go
	var a int64 = 123
	value := atomic.LoadInt64(&a)
	fmt.Println(value)
	
	atomic.StoreInt64(&a, 456)
	fmt.Println(a)
	
	// 对 a 进行 +1，并返回
	atomic.AddInt64(&a, 1)
	
	// 交换 a 和 789，并将a的旧值返回
	atomic.SwapInt64(&a, 789)
	
	// 比较并交换
	atomic.CompareAndSwapInt64(&a, 789, 123)
```

