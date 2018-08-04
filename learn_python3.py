
###############################
# 在实际编程中应当尽量避免isinstance()方法的使用，因为类好类型检查和python中多态的目标背道而驰
def my_abs(x : int):    # x : int 代表期望传入int类型参数
	if not isinstance(x, (int, float)):
		raise TypeError("bad operated type!")
	if x >= 0:
		return x
	else:
		return -x


print(my_abs(23), my_abs(-34))

###############################
# 位置参数、默认参数、命名关键字参数（字典形式存储）
def power(arg_1: int, n = 2, **temp) -> int:    # 方法期望返回值为int类型
	result = 1
	while n >0:
		result *= arg_1
		n -= 1
	print(temp)
	return result

    
print(power(5), power(5, 5, a = 3, b = 4))

###############################
# '*'为可变参数，job = 'Engineer'为关键字参数
def person(name, age, *, city='Beijing', job):
    print(name, age, city, job)


person('hezhihai', 23, job = 'Engineer')

###############################
# 使用尾递归求解阶乘
def fact(n: int)->int:
	return fact_iter(n, re_temp = 1)


def fact_iter(n_temp: int, re_temp: int):
	if n_temp == 1:
		return re_temp
	else:
		return fact_iter(n_temp - 1, re_temp * n_temp)


print(fact(10))

###############################
# 使用枚举enumerate
for i, value in enumerate('zhihai He'):
	print((i, value))

###############################
# 使用列表生成式
print(list(range(1, 20, 3)))
print([temp * temp for temp in range(1, 10)	if temp % 2 == 0])
print([m + n for m in 'ABC' for n in 'abc'])

###############################
import os
print([dirc for dirc in os.listdir()])
print((temp * temp for temp in range(1, 10)	if temp % 2 == 0))
# 构造生成器(generator)
s = (x * x for x in range(5))
print(s)
for x in s:
    print(x)

###############################
# 如果一个函数定义中包含yield关键字，那么这个函数就不再是一个普通函数，而是一个generator
def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a, b = b, a + b
        n = n + 1
    return 'done'


f = fib(10)
print('fib(10):', f)
for x in f:
    print(x)

###############################
# 使用Iterable
for x in 'zhihai He':
    print(x)

# 使用迭代器Iterator
it = iter('zhihai He')
while True:
    try:
        # 获取下一个值
        print(next(it))
    except StopIteration as e:
        print(e.value)
        break

###############################
f = abs
print(f(-21), f)

###############################
from collections import Iterable


# 使用lambda匿名函数
# map将传入的函数依次作用到序列的每个元素，返回结果为一个Iterator。
print(list(map(lambda x: x**2, range(1, 10, 2))))
print(isinstance(range(1, 10), Iterable))

###############################
def func(num):
    return num * num


it = map(func, list(range(1, 10, 2)))  # it为一个Iterator对象
# generator属于Iterator
# Iterator只能使用一次，故以下只会有一次输出
for i in list(it):
    print('This is ', i)
while True:
    try:
        print('And ', next(it))
    except StopIteration as e:
        print(e)
        break

###############################
from functools import reduce


# reduce函数对序列中的元素作累计效果运算，最后结果为一个值
def fun_num(num_1, num_2):
    return num_1 * 10 + num_2


def add(num_1, num_2):
    return num_1 + num_2


print(reduce(fun_num, [1, 2, 3, 4, 5]))
print(reduce(add, list(range(1, 10))))

###############################
# map对每一个迭代元素进行操作
# reduce把结果继续和序列的下一个元素做累积计算
from functools import reduce


def str2int(str_temp):
	dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
	return dic.get(str_temp)	


def fun_num(num_1, num_2):
	return num_1 * 10 + num_2


print(reduce(fun_num, map(str2int, '678')))

###############################
from functools import reduce


def my_sum(num_1, num_2):
	return num_1 + num_2


def sum_list(my_list):
	return reduce(my_sum, my_list)


print(sum_list(list(range(1, 100))))

###############################
from functools import reduce


def my_multi(num_1, num_2):
	return num_1 * 10 + num_2


def str2int(str_temp):
	dic = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
	return dic.get(str_temp)


def str2float(str_temp):
	int_list = str_temp.split('.')
	return reduce(my_multi, map(str2int, int_list[0])) + reduce(my_multi, map(str2int, int_list[1])) * pow (10, -1 * len(int_list[1]))


print('str2float(\'123.456\') =', str2float('123.456'))
if abs(str2float('123.456') - 123.456) < 0.00001:
    print('测试成功!')
else:
    print('测试失败!')

###############################
# 过滤器filter(function, iterable)实质为生成器(item for item in iterable if function(item))
def get_odd(int_num):
	return int_num % 2 == 0


print(list(filter(get_odd, list(range(1, 20)))))
print(list((item for item in range(1, 20) if item % 2 ==0)))
###############################
# lambda为匿名函数
from functools import reduce


print(reduce(lambda str_1, str_2: str_1 + str_2, map(str, map(lambda temp: temp * temp, range(1, 20, 3)))))

###############################
# strip()仅仅移除字符串头尾部条件字符
def not_empty(str_temp):
    return str_temp.strip()


print(list(filter(not_empty, [' zhihai He', '', ' ', 'He llo '])))

###############################
# 使用生成器生成1000以内所有奇数
def odd_iter():
	num = 1
	while True:
		num += 2
		yield num


for temp in odd_iter():
	if temp < 1000:
		print(temp)
	else:
		break

###############################
# 取所有奇数
def odd_iter():
    num = 1
    while True:
        num += 2
        yield num


def not_divisible(num_2):
    return lambda x: x % num_2 > 0


# 素数生成器
def primes():
    yield 2
    it = odd_iter()  # 初始化一个序列
    while True:
        n = next(it)
        yield n
        it = filter(not_divisible(n), it)   # 每次新创建一个过滤器


def print_primes(num):
    for temp in primes():
        if temp < num:
            print(temp)
        else:
            break


print_primes(100)

###############################
# 取回数
# 构造整数生成器
def num_iter():
    num = 1
    while True:
        yield num
        num += 1


def is_huishu(num):
    str_temp = str(num)
    if len(str_temp) <= 1:
        return True
    else:
        return str_temp[0] == str_temp[-1] and is_huishu(str_temp[1: -1])  # 使用递归判定是否为回数


def huishu(num):
    it = filter(is_huishu, num_iter())
    for temp in it:
        if temp < num:  # 结束条件
            print(temp)
        else:
            break


print(huishu(100000))

###############################
def num_iter():
	num = 1
	while True:
		yield num
		num += 1


def huishu(num):
	for temp in filter(lambda num: str(num) == str(num)[: : -1], num_iter()):
		if temp < num:
			print(temp)
		else:
			break


huishu(100)

###############################
# sorted(iterable, *, key=None, reverse=False)
print(sorted(list(range(30, 0, -2))))
print(sorted(list(range(30, 0, -2)), reverse=True))
# 分清函数调用func()与函数参数func
# 自定义排序，key指定的函数(映射)作用在每一个元素上
print(sorted([23, 9, -20, -1, 34], key=abs))
print(sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower))
print(sorted(['bob', 'about', 'Zoo', 'Credit'], key=str.lower, reverse=True))
dic = {1: 'f', 2: 'a', 3: 'z'}
# 字典默认按照关键字排序，key为一个映射
print(sorted(dic), sorted(dic, key=lambda a: dic[a]))
print(sorted([('Bob', 75, 's'), ('Adam', 92, 'a'), ('Bart', 66, 'b'), ('Lisa', 88, 'z')], key=lambda a: a[2]))

###############################
# 使用函数作为返回值，当一个函数返回了另一个函数后，其内部的局部变量还被新函数引用
# 使用闭包：返回闭包时牢记一点：返回函数不要引用任何循环变量，或者后续会发生变化的变量。
def count():
    i = 1

    def f(j):
        def g():
            nonlocal i
            i += 1  # 注意这里
            return j*j + i
        return g
    fs = []
    for i in range(1, 4):
        fs.append(f(i))     # f(i)立刻被执行，因此i的当前值被传入f()
    return fs


f1, f2, f3 = count()
print(f1, f2, f3)
print(f1(), f2(), f3())

###############################
from operator import itemgetter

students = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
print(sorted(students, key=itemgetter(0)))
print(sorted(students, key=itemgetter(1)))
print(sorted(students, key=lambda temp: temp[1]))
print(sorted(students, key=itemgetter(1), reverse=True))

###############################
# 将函数作为返回值
def lazy_sum(*num):
    def sum():
        sum_temp = 0
        for temp in num:
            sum_temp += temp
        return sum_temp

    return sum


# 每次调用都会返回一个新的函数，即使传入相同的参数:f1 != f2
# 当lazy_sum返回函数sum时，相关参数和变量都保存在返回的函数中
# 返回的函数并没有立刻执行，而是直到调用了f1()才执行
f1 = lazy_sum(1, 2, 3)
f2 = lazy_sum(2, 3, 4, 5, 5)
print(f1 == f2, f1(), f2())

###############################
funcs_1 = [lambda x, i=i: x*i for i in range(1, 10)]  # 生成列表中每个元素均为一个匿名函数
print(funcs_1[1], funcs_1[2](100))
funcs_2 = [lambda x: x*i for i in range(1, 10)]     # 在闭包中引用了循环变量
print(funcs_2[1], funcs_2[2](100))

###############################
# 返回的函数并没有立刻执行，而是直到调用了f1()才执行
# 结果全部都是9，原因就在于返回的函数引用了循环变量i，但它并非立刻执行。等到3个函数都返回时，它们所引用的变量i已经变成了3，因此最终结果为9。
# 返回闭包时牢记一点：返回函数不要引用任何循环变量，或者后续会发生变化的变量。
def count():
    fs = []
    for i in range(1, 4):
        def f():
            return i*i
        fs.append(f)
    return fs


f1, f2, f3 = count()
print(f1, f2, f3)
print(f1(), f2(), f3())

###############################
# 规避闭包中引用循环变量带来的错误
def count():
    fs = []
    for i in range(1, 4):
        def f(i=i):
            return i*i
        fs.append(f)
    return fs


f1, f2, f3 = count()
print(f1, f2, f3)
print(f1(), f2(), f3())

###############################
def count():
    def f(j):
        def g():
            return j*j
        return g
    fs = []
    for i in range(1, 4):
        fs.append(f(i))     # f(i)立刻被执行，因此i的当前值被传入f()
    return fs


f1, f2, f3 = count()
print(f1, f2, f3)
print(f1(), f2(), f3())

###############################
def count():
    fs = []
    for i in range(1,4):
        def f(j):   # 此处参数j无任何实际意义
            return lambda j: j*j    # 返回带参数函数
        fs.append(f(i))
    return fs


a, b, c = count()
print(a(0), a(1), a(2), a(3), a(4))
print(a(23), a(12), b(2), c(5))

###############################
def count():
    fs = []
    for i in range(1, 4):
        def f(j):
            return lambda: j * j
        fs.append(f(i))
    return fs


a, b, c = count()
print(a(), b(), c())

###############################
func = lambda x: x**2
print(func, func(23))

###############################
print(list(filter(lambda x: x % 2 == 1, range(1, 11))))

###############################
def my_func():
    print('called my_func()')


temp_func = my_func     # 相当于取别名
temp_func()
print(temp_func.__name__, my_func.__name__, temp_func == my_func)

###############################
# 在代码运行期间动态增加功能的方式，称之为“装饰器”（Decorator）
# 借助Python的@语法自定义装饰器，多个装饰器在应用时的顺序与指定顺序相反
# 无参数装饰器
import functools


def my_wrapper(my_func):
    @functools.wraps(my_func)
    def wrapper(*args, **kw):
        print('using my_wrapper.')
        return my_func(*args, **kw)
    return wrapper


@my_wrapper
def func():
    print('called my_func()')
    return 'called succeed!'


temp_func = func
print(temp_func.__name__, '\n', temp_func())

###############################
# 带参数装饰器
import functools


def my_wrapper(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print('calling my_wrapper, and your text is %s' % text)
            return func(*args, **kw)
        return wrapper
    return decorator


@my_wrapper('MindExp')
def my_func():
    return 'called my func()'


print(my_func.__name__)
print(my_func())

###############################
# 通过修改默认参数值，使用partial构造新的函数
import functools


int2 = functools.partial(int, base=2)
print(int2('10000'), int('10000', 2), int('10000'), int('10000', 8))
kw = {'base': 2}
print(int('1000', **kw))

###############################
import functools


def my_wrapper(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print('Helli, %s' % text)
            # 偏函数实质返回一个新函数
            # partial函数的作用就是把默认参数固定化，并返回一个固定化一部分参数后的函数
            return functools.partial(func, *args, **kw)		
        return wrapper
    return decorator


@my_wrapper('MindExp')
def my_func(n=10):
    print('This is %s.' % my_func.__name__)
    print('n = ', n)


my_func(100)()

###############################
import functools


def log(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print("%s begin call %s()" % (text, func.__name__))
            ret = func(*args, **kw)
            print("%s end call %s()" % (text, func.__name__))
            return ret
        return wrapper
    if isinstance(text, str):
        return decorator
    else:
        f = text
        text = ''
        return decorator(f)  # 对原函数f()使用装饰器


@log
def f():
    print("soso!")


f()

###############################
'''
每一个包目录下面都会有一个__init__.py的文件，这个文件是必须存在的，否则，Python就把这个目录当成普通目录，而不是一个包。
__init__.py可以是空文件，也可以有Python代码，因为__init__.py本身就是一个模块，而它的模块名就是mycompany
任何模块代码的第一个字符串都被视为模块的文档注释；
mycompany
 ├─ web
 │  ├─ __init__.py
 │  ├─ utils.py
 │  └─ www.py
 ├─ __init__.py
 ├─ abc.py
 └─ xyz.py
 '''
###############################
"""module doc"""  # 模块第一个字符串被视为模块文档注释
__author__ = 'MindExp'

import sys


print(sys.argv, '\n', sys.path, '\n', sys.modules)
print(__name__, __author__)

###############################
# object代表Student的父类
# 在类中定义的函数第一个参数必须为self，其他特性与普通函数一致(可使用位置参数，默认参数，变长参数，关键字参数，命名关键字参数)
class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def print_student_info(self):
        print('%s: %s' % (self.name, self.score))


zhihai = Student('zhihai He', 668)
zhihai.print_student_info()

###############################
class Student(object):
    # python中内置许多特殊变量，以'__'开头，并以'__'结尾，usage: __main__
    # 在类中以'__'开头的变量为private变量
    # 在类中以'_'开头的为私有变量，但是在类外部是可以访问的，即：“虽然我可以被访问，但是，请把我视为私有变量，不要随意访问”
    def __init__(self, stu_id, stu_name):
        self.__stu_id = stu_id
        self.__stu_name = stu_name

    def print_stu_info(self):
        print('tu_id = %s, stu_name = %s' % (self.__stu_id, self.__stu_name))


bruce = Student('201722060825', 'zhihai He')
bruce.print_stu_info()
# python没有任何机制阻止coder干坏事，一切靠自觉
# 试图修改private变量，但是不会成功
bruce.__stu_id = '668'
print(bruce.__stu_id, bruce._Student__stu_id)   # 此时’__stu_id‘和对象中的'__stu_id'（实质被当前解释器解释为：_Student__stu_id）并不是同一个变量
bruce.print_stu_info()
# 试图修改private变量，能成功，但是强烈建议不要这么做，因为在不同版本python解释器中__stu_id会被改成不同的变量名
bruce._Student__stu_id = '668'
bruce.print_stu_info()

###############################
class Animal(object):
    def run(self):
        print('Animal is running.')

    def __len__(self):
        return 100

    def len(self):
        return 10

class Cat(Animal):  # 使用继承
    def run(self):  # 重写父类run()函数
        print('Cat is running.')

    def voice(self):
        print('miaomiao.')

class Car(object):
    def run(self):  # 重写父类run()函数
        print('Car is running.')

def run_twice(animal):
    animal.run()    # 多态
    animal.run()


cat = Cat()
cat.run()
cat.voice()
animal = Animal()
print(isinstance(cat, Animal), isinstance(animal, Cat))
run_twice(animal)
run_twice(cat)
# python为动态、非严格继承语言，鸭子类型。
# 一个对象只要“看起来像鸭子，走起路来像鸭子”，那它就可以被看做是鸭子
car = Car()
run_twice(car)
# 能用type()判断的基本类型也可以用isinstance()判断
print(type(123), type('123'), type(12.3), type(cat), type(animal), type(car))
# 使用dir(object)函数获取给定对象所有的属性和方法
print(dir(cat))
print(len(cat))  # 实质调用系统中__len__()函数
print(cat.len(), cat.__len__())

###############################
class Animal(object):
    def run(self):
        print('Animal is running.')

    def __len__(self):
        return 100

class Car(object):
    def run(self):
        print('Car is running.')


animal = Animal()
car = Car()
print(hasattr(car, 'type'), hasattr(car, 'price'), hasattr(car, 'y'))
# 设置对象属性值方式有三种
# object.name = value, setattr(object, name, value), object.__setattr__(name, value)，
car.type = 'AL6'
setattr(car, 'price', '46.9')
car.__setattr__('color', 'red')
# 获取属性值的方式有三种
# object.name, getattr(object, name), object.__getattribute__(name)
print(car.type, getattr(car, "price"), car.__getattribute__('color'))
# 在不带括号运算中，and运算符优先级大于or运算符优先级
# 在and运算符中（statement_1 and statement_2）如果，statement_1为真，则返回statement_2，否则返回statement_1
# 在or运算符中（statement_1 or statement_2）如果，statement_1为真，则返回statement_1，否则返回statement_2
print((tuple and list), (tuple or list), isinstance([1, 2, 3], (tuple and list)))
print((list and tuple), (list or tuple), isinstance([1, 2, 3], (list and tuple)))

###############################
# 类属性与实例属性
class Student(object):
    # 类属性，所有类实例均可以共享类属性
    name = 'Student'
    count = 0

    def __init__(self):
        Student.count += 1

def get_name(self):
    return self.name

def my_private_func():
    return 'My private function called.'


# 绑定类方法
Student.get_name = get_name
bruce = Student()
print(bruce.get_name(), bruce.count, Student.name, Student.count)
# 添加实例属性，对类属性无影响，但实例属性优先级高于类属性
bruce.name = 'Bruce'
jack = Student()
# 使用bruce.name时类属性不可见
print(jack.get_name(), bruce.get_name(), Student.name, Student.count)
# 删除实例属性
del bruce.name
print(jack.get_name(), bruce.get_name(), Student.name, Student.count)
# 绑定对象方法
jack.my_private_func = my_private_func
print(jack.my_private_func())
try:
    print(bruce.my_private_func())
except AttributeError:
    raise AttributeError("Object function has not bound.")

###############################
class Student(object):
    __slots__ = ('name', 'age')
# 使用__slots__特殊变量限制类实例随意添加类属性，但是对继承他的子类不起作用
# 除非在子类中也定义__slots__特殊变量，这样，子类实例允许定义的属性就是自身的__slots__加上父类的__slots__


bruce = Student()
bruce.name = 'Bruce'
bruce.age = 23
try:
    bruce.address = 'Sichuan Province'
except AttributeError:
    raise AttributeError('Binding attribute failed.')
print(dir(bruce))

###############################
    class Student(object):
    __slots__ = ('name', 'age', 'score')

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def set_name(self, name):
        self.name = name

    # 通过python内置的property装饰器将方法变成属性调用
    @property   # 定义getter
    def score_info(self):
        print("Do something in getter!")
        return self.score   # 在作为setter时，此处并无实际意义，但如果直接使用object.set_score，而不是作为赋值语句会导致overflow

    # 使用装饰器对setter进行数值检查
    @score_info.setter   # 定义setter（需要一个property函数对象）
    def score_info(self, score):
        print("Do something in setter!")
        if not isinstance(score, int):
            raise ValueError('Score must be an integer.')
        if score < 0 or score > 100:
            raise ValueError('Score must between 0 ~ 100.')
        self.score = score

    def get_name(self):
        return self.name

    def get_age(self):
        return self.age


bruce = Student('Bruce', 24)
print(bruce.get_name(), bruce.get_age())
# set_score()方法当属性使用，通过装饰器进行数值检查
bruce.score_info = 100
print(bruce.score_info, bruce.score)

###############################
class Student(object):
    # 使用__slots__后无法在外部直接访问private属性(__age)
    __slots__ = ('__name', '__age', '__score', '_t')

    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    def name(self, name):
        self.__name = name

    # 通过python内置的property装饰器将方法变成属性调用
    # 定义getter
    @property
    def score(self):
        return self.__score

    # 定义setter
    @score.setter
    def score(self, score):
        if not isinstance(score, int):
            raise ValueError('Score must be an integer.')
        if score <0 or score >100:
            raise ValueError('Score must between 0 ~ 100.')
        self.__score = score

    # 只定义getter方法，不定义setter方法就是一个只读属性
    # 定义getter
    @property
    def name(self):
        return self.__name

    # 定义getter
    @property
    def age(self):
        return self.__age


bruce = Student('Bruce', 24)
print(bruce.name)
# set_score当属性使用，通过装饰器进行数值检查
bruce.score = 100
print(bruce.score)
try:
    # __age为对象限定属性(__slots__)，故无法在外部访问，即使是形式访问也不行
    bruce.__age = 100
except AttributeError:
    raise AttributeError('Student object has no attribute __age.')  # 抛出AttributeError错误

###############################
# Python允许使用多重继承，因此，MixIn就是一种常见的设计
class Student(object):
    def __init__(self, name):
        self.name = name

    # 定义__str__()方法后，打印对象时直接调用__str__()方法
    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Hello,__repr__().'


# 匿名对象
print(Student('Bruce'))  # 默认自动调用__str__函数
Student.__str__ = Student.__repr__
jack = Student('Jack')
print(jack)

###############################
# 定义可迭代对象，主要实现一个对象可迭代化__iter__()并实现__next__()方法
class Fib(object):
    def __init__(self):
        self.a, self.b = 1, 1

    def __iter__(self):
        return self

    def __next__(self):
        self.a, self.b = self.b, self.a + self.b
        if self.a > 10000:
            raise StopIteration()
        return self.a

    def __getitem__(self, item):
        # 按照下标取出元素
        if isinstance(item, int):
            i, j = 1, 1
            for x in range(item):
                i, j = j, i + j
            return i
        # 可迭代对象切片
        if isinstance(item, slice):
            start = item.start
            stop = item.stop
            if start is None:
                start = 0
            L = []
            i, j = 1, 1
            for x in range(stop):
                if x >= start:
                    L.append(i)
                i, j = j, i + j
            return L


# 实例化一个
fib = Fib()
for i in fib:
    print(i)
print(fib[9])
print(fib[5: 10])

###############################
class Student(object):
    def __init__(self, name):
        self._name = name

    # 将name方法属性化
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        # 一定不能使用self.name = name.会出现无穷递归，因为此时name方法已属性化.
        self._name = name
        return 

    # 当实例化对象试图获取未定义属性时，会自动调用__getattr__方法
    def __getattr__(self, attr):
        if attr == 'nick_name':
            return 'MindExp'
        if attr == 'func':
            return lambda: 100
        raise AttributeError('Student has no attribute %s.' % attr)


bruce = Student('zhihai He')
bruce.name = 'Bruce'
print(bruce.name)
print(bruce.nick_name)
print(bruce.func())
print(bruce.address)

###############################
class RestApi(object):
    def __init__(self, path):
        self._path = path

    # 当实例化对象试图获取未定义属性时，会自动调用__getattr__方法，从而把一个类的所有方法和属性都动态化处理
    # 利用__getattr__方法实现链式调用
    def __getattr__(self, path):
        return RestApi('%s/%s' % (self._path, path))

    # 调用对象时会自动调用__str__方法
    def __str__(self):
        return self._path

    __repr__ = __str__


print(RestApi('').a.f.r)
print(RestApi('MindExp').g.h)

###############################
class Student(object):
    def __init__(self, name):
        self.name = name

    # 实现对象可调用化，即可以把一个对象看做方法，由于方法在运行期间创建，故类的实例在运行期间创建，从而模糊了对象和方法的界限
    def __call__(self, *args, **kwargs):
        return '%s is callable.' % self.name

    def __str__(self):
        return 'Call me baby.'

    __repr__ = __str__


bruce = Student('Bruce')
print(callable(bruce), callable(max))
print(bruce, str(bruce))
# 调用类中__call__方法
print(bruce())

###############################
from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    BLACK = 'black'


print(Color.RED, str(Color.RED), repr(Color.RED), type(Color.RED))
print(isinstance(Color.RED, Color), isinstance(Color.RED, Enum), isinstance(Color, Enum), type(Color), type(Enum))
print(Color.RED.name, Color.RED.value)
# Enumerations support iteration
for color in Color:
    print(color.name, color.value)
print(Color['BLACK'], Color(2), Color(2).name)

###############################
def func(self, name='MindExp'):
    print('Hello, %s' % name)


# 使用type动态创建类,，type(name, bases, dict)
Hello = type('Hello', (object,), dict(hello=func))
h = Hello()
h.hello()
print(type(Hello), type(h), h, h.hello)

###############################
# 参考资料：http://blog.jobbole.com/21351/
# 元类用来创建类（对象）。即：元类为类的类
"""
MyClass = MetaClass()
my_object = MyClass()

MyClass = type(MyClass, (), {})
type就是Python在背后用来创建所有类的元类
"""
# 元类会自动将你通常传给'type'的参数作为自己的参数传入
# 注：以下程序在python3.6环境下测试失败
def upper_attr(future_class_name, future_class_parents, future_class_attr):
    """
    返回一个类对象，将属性都转为大写形式
    """
    #  选择所有不以'__'开头的属性
    attrs = ((name, value) for name, value in future_class_attr.items() if not name.startswith('__'))
    # 将它们转为大写形式
    uppercase_attr = dict((name.upper(), value) for name, value in attrs)
    # 通过'type'来做类对象的创建
    return type(future_class_name, future_class_parents, uppercase_attr)


__metaclass__ = upper_attr  # 这会作用到这个模块中的所有类


class Foo(object):
    # 我们也可以只在这里定义__metaclass__，这样就只会作用于这个类中
    bar = 'bip'


print(hasattr(Foo, 'bar'))
# 输出: False
print(hasattr(Foo, 'BAR'))
# 输出:True
f = Foo()
print(f.BAR)
# 输出:'bip'

###############################
import logging


# 使用logging，有debug，info，warning，error等几个级别，最后统一控制输出哪个级别的信息
logging.basicConfig(level=logging.INFO)     # 指定记录信息的级别
try:
    pass
# 捕获错误信息
except ValueError:
    # 抛出错误信息,停止运行
    raise ValueError('ValueError.')
    pass
except ZeroDivisionError as e:
    # 记录错误信息，并让程序继续执行
    logging.ZeroDivisionError(e)
    pass
# 没有错误发生时，自动执行else语句
else:
    pass

###############################
# TDD：Test-Driven Development
# py_01.py
class Dict(dict):
    def __init__(self, **kw):
        super().__init__(**kw)

        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(r"'Dict' object has no attribute '%s'" % key)

        def __setattr__(self, key, value):
            self[key] = value



# py_02.py
import unittest
from py_01 import Dict


class TestDict(unittest.TestCase):
    # 每调用一个测试方法前执行（启动数据库连接）
    def setUp(self):
        print('setUp...')

    def test_init(self):
        d = Dict(a=1, b='test')
        # 断言函数返回的结果与1相等
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 'test')
        self.assertTrue(isinstance(d, dict))

    def test_key(self):
        d = Dict()
        d['key'] = 'value'
        self.assertEqual(d.key, 'value')

    def test_attr(self):
        d = Dict()
        d.key = 'value'
        self.assertTrue('key' in d)
        self.assertEqual(d['key'], 'value')

    def test_keyError(self):
        d = Dict()
        # 断言抛出期待指定类型的错误
        with self.assertRaises(KeyError):
            value = d['empty']

    def test_arrtError(self):
        d = Dict()
        with self.assertRaises(AttributeError):
            value = d.empty

    # 每调用一个测试方法后执行（关闭数据库连接）
    def tearDown(self):
        print('tearDown...')


if __name__ == '__main__':
    # 单元测试
    unittest.main()

###############################
import doctest


class Dict(dict):
    # doctest严格按照Python交互式命令行的输入和输出来判断测试结果是否正确。只有测试异常的时候，可以用...表示中间一大段烦人的输出。
    """
    Simple dict but also support access as x.y style.
    >>> d1 = Dict()
    >>> d1['x'] = 100
    >>> d1.x
    100
    >>> d1.y = 200
    >>> d1['y']
    200
    >>> d2 = Dict(a=1, b=2, c='3')
    >>> d2.c
    '3'
    >>> d2['empty']
    Traceback (most recent call last):
        ...
    KeyError: 'empty'
    >>> d2.empty
    Traceback (most recent call last):
        ...
    AttributeError: 'Dict' object has no attribute 'empty'
    """

    def __init__(self, **kw):
        super(Dict, self).__init__(**kw)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(r"'Dict' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == '__main__':
    # 文档测试
    doctest.testmod()

###############################
import re


# 正则表达式也是用字符串表示的，强烈建议使用Python的r前缀，就不用考虑转义的问题了
"""
>>> import re
>>> re.match(r'^\d{3}\-\d{3,8}$', '010-12345')
<_sre.SRE_Match object; span=(0, 9), match='010-12345'>
>>> re.match(r'^\d{3}\-\d{3,8}$', '010 12345')
>>>

>>> re.split(r'[\s\,\;]+', 'a,b;; c  d')
['a', 'b', 'c', 'd']

# 用()表示的就是要提取的分组group
>>> m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
>>> m
<_sre.SRE_Match object; span=(0, 9), match='010-12345'>
# 注意group(0)永远是原始字符串，group(1)、group(2)……表示第1、2、……个子串
>>> m.group(0)
'010-12345'
>>> m.group(1)
'010'
>>> m.group(2)
'12345'

# 贪婪匹配
>>> re.match(r'^(\d+)(0*)$', '102300').groups()
('102300', '')
# 非贪婪匹配
>>> re.match(r'^(\d+?)(0*)$', '102300').groups()
('1023', '00')

>>> import re
# 预编译,编译后生成Regular Expression对象，由于该对象自己包含了正则表达式，所以调用对应的方法时不用给出正则字符串
>>> re_telephone = re.compile(r'^(\d{3})-(\d{3,8})$')
# 使用：
>>> re_telephone.match('010-12345').groups()
('010', '12345')
>>> re_telephone.match('010-8086').groups()
('010', '8086')

"""
# match()方法判断是否匹配，如果匹配成功，返回一个Match对象，否则返回None
test = '用户输入的字符串'
if re.match(r'正则表达式', test):
    print('ok')
else:
    print('failed')

###############################
class Bird(object):
    def __init__(self):
        self.hungry = True

    def eat(self):
        if self.hungry:
            print("Aaaah")
            self.hungry = False
        else:
            print("No, Thanks!")


class SongBird(Bird):
    # 重写__init__方法
    def __init__(self):
        """
        使用父类__init__两种方法如下：
        Bird.__init__(self)     # 旧版python方法
        super(SongBird, self).__init__()    #新版python中使用super方法
        """
        super(SongBird, self).__init__()
        self.sund = "Squawk"

    def sing(self):
        print(self.sund)


sb = SongBird()
sb.sing()
sb.eat()
sb.eat()

###############################
class Rectangle(object):
    def __init__(self):
        self.width = 0
        self.height = 0

    def setSize(self, size):
        self.width, self.height = size

    def getSize(self):
        return self.width, self.height

    # property函数的使用，类似提供size接口，无需关心getSize, setSize方法具体实现
    size = property(getSize, setSize)


r = Rectangle()
r.width = 10
r.height = 5
print(r.getSize(), r.size)
r.size = (100, 50)
print(r.getSize(), r.size)

###############################
class MyClass(object):
    def __init__(self):
        return "object created!"

    # 定义静态方法
    @staticmethod
    def smethod():
        print("smethod called.")

    # 定义类方法
    @classmethod
    def cmethod(cls):
        print("cmethod called.")


MyClass.smethod()
MyClass.cmethod()

###############################
class MyClass(object):
    def __init__(self):
        print("__init__ been called.")

    # 获取item特性时被调用
    def __getattribute__(self, item):
        print("__getattribute__ been called.")
        try:
            return object.__getattribute__(self, item)
        except:
            print("AttributeError")

    # 试图给item特性赋值时被调用
    def __setattr__(self, key, value):
        print("__setattr__ been called.")
        object.__setattr__(self, key, value)


myObject = MyClass()
myObject.info
myObject.info = 'info'
print(myObject.info)

###############################
# 使用递归生成器
def flatten(nested):
    try:
        for sublist in nested:
            for element in flatten(sublist):
                yield element
    except TypeError:
        yield nested


print(list(flatten([[[[1, 2], 3], 4], 5])))

###############################
import random

# 冲突检查，在定义state时，采用state来标志每个皇后的位置，其中索引用来表示横坐标，基对应的值表示纵坐标，例如： state[0]=3，表示该皇后位于第1行的第4列上
def conflict(state, nextX):
    nextY = len(state)
    for i in range(nextY):
        # 如果下一个皇后的位置与当前的皇后位置相邻（包括上下，左右）或在同一对角线上，则说明有冲突，需要重新摆放
        if abs(state[i] - nextX) in (0, nextY - i):
            return True
    return False


# 采用生成器的方式来产生每一个皇后的位置，并用递归来实现下一个皇后的位置。
def queens(num, state=()):
    for pos in range(num):
        if not conflict(state, pos):
            # 产生当前皇后的位置信息
            if len(state) == num - 1:
                yield (pos,)
            # 否则，把当前皇后的位置信息，添加到状态列表里，并传递给下一皇后。
            else:
                for result in queens(num, state + (pos,)):
                    yield (pos,) + result


# 为了直观表现棋盘，用X表示每个皇后的位置
def prettyPrint(solution):
    def line(pos, length=len(solution)):
        return '. ' * (pos) + 'X ' + '. ' * (length - pos - 1)

    for pos in solution:
        print(line(pos))


if __name__ == "__main__":
    prettyPrint(random.choice(list(queens(8))))

###############################
from heapq import *
from random import shuffle


"""
1. 在python中并没有独立的堆类型，只有一个包含一些堆操作函数的模块，这个模块叫做heapq
2. 在python堆底层算法实现中，i位置处的元素总比2*i以及2*i+1位置处的元素小，该特性称为堆属性
3. heapify函数使用任意列表作为参数，并通过尽量少的以为操作，将其转换为合法的堆（满足堆属性）
"""
data = list(range(10))
shuffle(data)
heap = []   # heap = heapify(data)
for n in data:
    heappush(heap, n)
print(heap)
heappush(heap, 0.5)
print(heap)
heappop(heap)
print(heap)     # 弹出最小元素，一般都是索引位置为0的元素
heapreplace(heap, 66)
print(heap)

###############################
from collections import deque


# 创建双端队列
queue = deque(range(10))
queue.append(66)
queue.appendleft(88)
print(queue)
print(queue.pop())
print(queue.popleft())
queue.rotate(3)
print(queue)
queue.rotate(-1)
print(queue)

###############################
import time


startTime = time.time()     # 获取当前时间（新纪元开始后的秒数，以UTC为准）
print(time.asctime())   # 将当前时间转换为字符串
time.sleep(3)   # 休眠3秒
endTime = time.time()
print(endTime - startTime)

###############################
from random import *


print(random())     # 生成（0，1）之间的随机实数
print(uniform(10, 20))      # 生成（10,20）之间的随机实数
print(randrange(20, 40, 5))     # 随机返回生成列表中的数
print(choice(list(range(20, 40, 5))))   # 随机返回生成列表中的数
print(sample(list(range(20, 40, 5)), 3))    # 在列表中随机选择3个独立元素

###############################
import sys, shelve


"""
使用shelve作为临时存储方案（实际在文件中以字典形式存储）
"""
def store_person(db):
    pid = input('Enter unique ID number:')
    person = {'name': input('Enter name:'), 'age': input('Enter age:'), 'phone': input('Enter phone number:')}
    db[pid] = person


def lookup_person(db):
    pid = input('Enter ID number:')
    field = input('What would you like to know? (name, age, phone, all info)')
    field = field.strip().lower()
    if field == 'all info':
        print(field.capitalize() + ':' + str(db.get(pid, 'No looking up info in database.dat file')))
    else:
        print(field.capitalize() + ':' +
              db.get(pid, 'No looking up info in database.dat file')(field, 'No looking up info in database.dat file'))
        db.pop()

def print_help():
    print('Help information in here.')


def enter_command():
    cmd = input('Enter command(? for help):')
    cmd = cmd.strip().lower()
    return cmd


def main():
    database = shelve.open('F:\\Projects\\Python\\Py_01\\database.dat')
    try:
        while True:
            cmd = enter_command()
            if cmd == 'store':
                store_person(database)
            elif cmd == 'lookup':
                lookup_person(database)
            elif cmd == 'help':
                print_help()
            else:
                return
    finally:
        database.close()


if __name__ == '__main__':
    main()

###############################
"""
import re
Simple Regular Expression Syntax
通配符（.），可以匹配除了换行符外的任意字符。如：'.ython'
对特殊字符需要进行转义。如：'python\\.org'    # 用到了两层转义
字符集：'[a-zA-Z0-9]'能匹配任意大小写字母和数字。
反转字符集：'[^abc]'能匹配除了a、b和c之外的字符。
选择符：'pythob|perl'
子模式：'p(ython|erl)'
可选项：在子模式后面加上问号表示可选项（出现一次或者不出现），如：r'(http://)?(www\.)?python\.org'
(pattern)*:模式重复出现0次或者多次
(pattern)+：模式重复出现1次或者多次
(pattern){m,n}：模式重复出现m到n次
字符串的开始与结尾：使用脱字符(^)标记开始匹配，如：'^ht+p'

re模块中的一些重要函数
compile(pattern[, flags])   # 创建模式对象
search(pattern, string[, flags])    # 在字符串中寻找模式
match(pattern, string[, flags])     # 在字符串开始处匹配模式
split(pattern, string[, maxsplit=0])    # 根据模式匹配项来分割字符串
findall(pattern, string)    #列出字符串中所有匹配项
sub(pat, repl, string[, count=0])   # 将字符串中所有pat项用repl替换
escape(string)  # 将字符串中所有特殊正则表达式字符转义    
"""

###############################
someFile = open(r'F:\Projects\Python\Py_01\myfile.txt', 'r+', True)
# python中针对文件通用操作步骤
try:
    someFile.doSomething()
except Exception as e:
    print(e)
finally:
    someFile.close()

# python中针对文件操作的with语句，在with语句结束时会自动close文件
with open('', 'r', True) as someFile:
    someFile.doSomething()

###############################
# 对文件内容进行迭代
def process(string):
    print('Processing' + string)


# method one
file = open('File path')
char = file.read(1)
while char:
    process(char)
    char = file.read(1)
# method two
while True:
    char = file.read(1)
    if not char:
        break
    process(char)
# 按行读取内容
while True:
    line = file.readline()
    if not line:
        break
    process(line)
# 迭代字符：file.read()会读取文件所有内容，返回字符串。
for char in file.read():
    process(char)
# 迭代行：file.readlines()也会读取所有内容，但是返回列表
for line in file.readlines():
    process(line)
file.close()

###############################
import fileinput


def process(string):
    print('Processing: ' + string)


file = open('F:\Projects\Python\Py_01\myfile.txt', 'r')    # 默认read only方式打开
for line in fileinput.input(file):  # 测试无效，无法迭代
    process(line)
# 文件迭代器
for line in file:   # 文件迭代对象实际为内容列表
    process(line)
file.close()

###############################
    Core Python Programming
###############################
import re

# 正则表达式对于探索原始字符串有着强大的动力，原因就在于 ASCII 字符和正则表达式的特殊字符之间存在冲突
m = re.match('foo', 'sea food')  # 模式匹配字符串，匹配成功就返回匹配对象，否则返回None
if m is not None:  # 如果匹配成功，就使用group()输出匹配内容
    print(m.group())
else:
    print('mach failed.')
print(re.search('foo', 'sea food').group())
pattern = '\w+@(\w+\.)?\w+\.com'  # 使用圆括号对正则表达式进行分组（子组）
print(re.match(pattern, 'nobody@xxx.com').group())
print(re.match(pattern, 'nobody@www.xxx.com').group())
pattern = '(\w+)-(\d+)'
m = re.match(pattern, 'abc-123')
if m is not None:
    print(m.groups(), m.group(), m.group(1), m.group(2))
m = re.match('(a(b))', 'ab')
if m is not None:
    print(m.groups(), m.group(), m.group(1), m.group(2))
print(re.search('^The', 'The end.').group())
m = re.search('^The', 'end. The')
if m is not None:
    print(m.group())
else:
    print('search failed.')
print(re.search(r'\bthe', 'bite the dog.').group())  # 使用原始字符串
m = re.search(r'\bthe', 'bitethe dog.')  # '\b'表示匹配任何单词边界，'\B'反之
if m is not None:
    print(m.group())
else:
    print('search failed.')
print(re.search(r'\Bthe', 'bitethe dog.').group())
print(re.findall('car', 'carry the barcardi to the car.'))  # findall()以列表形式返回匹配结果
string = 'This and that'
m1 = re.findall(r'(th\w+) and (th\w+)', string, re.I)  # findall()匹配成功时以列表形式返回，其中re.I 表示匹配时忽略大小写
it = re.finditer(r'(th\w+) and (th\w+)', string, re.I)
print(m1)
print(next(it).groups())
print(re.findall(r'(th\w+)', string, re.I))
it = re.finditer(r'(th\w+)', string, re.I)
print(next(it).groups())
print(next(it).groups())
print(re.sub(r'X', 'Mr.Smith', 'attn: X\n\tDear X, \n'))  # 搜索和替换字符串
print(re.subn(r'X', 'Mr.Smith', 'attn: X\n\tDear X, \n'))  # 搜索和替换字符串，并返回替换总数，以元组形式返回
print(re.split(':', 'str1:str2:str3'))  # 以列表形式返回
# 使用扩展符（使用(?iLmsux)系列选项）
print(re.findall(r'(?i)yes', 'yes? Yes, YES!!'))  # 扩展符(?i)表示忽略大小写
print(re.findall(r'th.+', """
    The first line
    the second line
    the third line"""))  # '.'无法匹配换行符
print(re.findall(r'(?s)th.+', """
    The first line
    the second line
    the third line"""))  # 扩展符(?s)表示'.'可以匹配换行符
print(re.findall(r'http://(?:\w+\.)*(\w+\.com)', 'http://google.com http://www.google.com http: '
                                                 '//code.google.com'))  # 扩展符(?...)对部分正则表达式进行分组，但是并不会保存该分组用于后续的检索或者应用
print(re.search(r'\((?P<areacode>\d{3})\) (?P<prefix>\d{3})-(?:\d{4})',
                '(800) 555-1212').groupdict())
# 正则表达式本质上实现贪婪匹配
print(re.match(r'.+(\d+-\d+-\d+)', 'Thu Feb 15 17:46:04 2007::uzifzf@dpyivihw.gov::1171590364-6-8').group(1))
# 使用“非贪婪”操作符“?”
print(re.match(r'.+?(\d+-\d+-\d+)', 'Thu Feb 15 17:46:04 2007::uzifzf@dpyivihw.gov::1171590364-6-8').group(1))

###############################
from socket import *


tcpSockt = socket(AF_INET, SOCK_STREAM)    # 创建TCP套接字
udpSockt = socket(AF_INET, SOCK_DGRAM)     # 穿件UDP套接字
'''
TCP服务器通用设计模式：
    ss = socket()  # 创建服务器套接字
    ss.bind()  # 套接字与地址绑定
    ss.listen()  # 监听连接
    iif_loop:  # 服务器无限循环
        cs = ss.accept()  # 接受客户端连接
        comm_loop:  # 通信循环
            cs.recv()/cs.send()  # 对话（接收/发送）
        cs.close()  # 关闭客户端套接字
    ss.close()  # 关闭服务器套接字#（可选）

TCP客服端通用设计模式：
    cs = socket()
    cs.connect()
    comm_loop:
        cs.send()/cs.recv()
    cs.close()

UDP服务器通用设计模式：
    ss = socket()
    ss.bind()
    iif_loop:
        ss.recvfrom()/ss.sendto()
    ss.close()
UDP客服端通用设计模式：
    cs = socket()
    comm_loop:
        cs.sendto()/cs.revefrom()
    cs.close()
'''

###############################
# 创建TCP服务端与TCP客户端并实现简单的相互通信
# TCP服务端
from socket import *
from time import ctime

# 创建一个 TCP 服务器，它接受来自客户端的消息，然后将消息加上时间戳前缀并发送回客户端
HOST = 'localhost'
PORT = 21567
BUFSIZE = 1024
ADDRESS = (HOST, PORT)

tcpServerSocket = socket(AF_INET, SOCK_STREAM)
tcpServerSocket.bind(ADDRESS)
tcpServerSocket.listen(5)  # 在连接被转接或拒绝之前，传入连接请求的最大数
try:
    while True:
        print('waiting for connection...')
        tcpClientSocket, address = tcpServerSocket.accept()     # 面向连接的，需建立虚电路连接
        print('... connected from:', address)

        while True:
            data = tcpClientSocket.recv(BUFSIZE)
            if not data:
                break
            print('Server received data: %s ' % (data.decode('utf-8')))
            tcpClientSocket.send(bytes('[%s] %s' % (ctime(), data), 'utf-8'))  # 将字符串作为一个 ASCII 字节“字符串”发送

        tcpClientSocket.close()
except (EOFError, KeyboardInterrupt) as e:
    print(e)
finally:
    tcpServerSocket.close()

###############################
# TCP客户端
from socket import *

HOST = 'localhost'
PORT = 21567
BUFFSIZE = 1024
ADDRESS = (HOST, PORT)

tcpClientSocket = socket(AF_INET, SOCK_STREAM)
tcpClientSocket.connect(ADDRESS)

while True:
    data = input('>')
    if not data:
        break
    tcpClientSocket.send(bytes(data, 'utf-8'))
    data = tcpClientSocket.recv(BUFFSIZE)
    if not data:
        break
    print(data.decode('utf-8'))

tcpClientSocket.close()

###############################
# UDP服务端
from socket import *
from time import ctime

HOST = 'localhost'
PORT = 21567
BUFFSIZE = 1024
ADDRESS = (HOST, PORT)

udpServerSocket = socket(AF_INET, SOCK_DGRAM)
udpServerSocket.bind(ADDRESS)

try:
    while True:
        print('waiting for message...')
        data, address = udpServerSocket.recvfrom(BUFFSIZE)
        print('...received from and returned to (%s, %s)' % address)
        print('Server received data: %s ' % (data.decode('utf-8')))
        udpServerSocket.sendto(bytes('[%s] %s' % (ctime(), data), 'utf-8'), address)
except (EOFError, KeyboardInterrupt) as e:
    print(e)
finally:
    udpServerSocket.close()

###############################
# UDP客户端
from socket import *

HOST = 'localhost'
PORT = 21567
ADDRESS = (HOST, PORT)
BUFFSIZE = 1024

udpClientSocket = socket(AF_INET, SOCK_DGRAM)

while True:
    data = input('>')
    if not data:
        break
    udpClientSocket.sendto(bytes(data, 'utf-8'), ADDRESS)
    data, ADDRESS = udpClientSocket.recvfrom(BUFFSIZE)
    if not data:
        break
    print(data.decode('utf-8'))

###############################
# socketserver TCP服务器
from socketserver import (TCPServer as TCP,
                          StreamRequestHandler as SRH)
from time import ctime

# 创建socketserver TCP服务器，其为标准库中的一个高级模块，主要使用大量的类隐藏了实现细节
HOST = 'localhost'
PORT = 21567
ADDRESS = (HOST, PORT)

class MyRequestHandler(SRH):
    def handle(self):   # 重写StreamRequestHandler中handle()
        print('...connected from: (%s, %s)' % self.client_address)
        # StreamRequestHandler类将输入和输出套接字看作类似文件的对象
        data = self.rfile.readline().decode()
        print('received data: %s ' % data)
        self.wfile.write(bytes('[%s] %s' % (ctime(), data), 'utf-8'))


tcpServer = TCP(ADDRESS, MyRequestHandler)
print('waiting for connection...')
tcpServer.serve_forever()

###############################
# TCP客服端
from socket import *

HOST = 'localhost'
PORT = 21567
BUFFZISE = 1024
ADDRESS = (HOST, PORT)

while True:
    tcpClientSocket = socket(AF_INET, SOCK_STREAM)
    tcpClientSocket.connect(ADDRESS)
    data = input('>')
    if not data:
        break
    tcpClientSocket.send(bytes('%s\r\n' % data, 'utf-8'))
    data = tcpClientSocket.recv(BUFFZISE)
    if not data:
        break
    print(data.decode().strip())
    tcpClientSocket.close()

###############################
# twisted框架：TCP服务端
from twisted.internet import protocol, reactor
from time import ctime

# 创建 Twisted Reactor TCP 服务器
PORT = 21566
# 创建协议类，并实现异步操作
class TSServerProtocol(protocol.Protocol):
    def connectionMade(self):
        host = self.host = self.transport.getPeer().host
        port = self.port = self.transport.getPeer().port
        print('...connected from:(%s, %s)' % (host, port))

    def dataReceived(self, data):
        print('received data: %s' % data.decode('utf-8'))
        self.transport.write('[%s] %s' % (ctime(), data))


factory = protocol.Factory()    # 每次接到一个连接时，“制造”一个协议实例
factory.protocol = TSServerProtocol()     # 接收到一个请求时，就会创建一个 TSServProtocol 实例来处理那个客户端的事务
print('waiting for connection...')
reactor.listenTCP(PORT, factory)    # 在reactor中安装一个TCP监听器，以此检查服务请求
reactor.run()

###############################
# twisted框架：TCP客服端
from twisted.internet import protocol, reactor

HOST = 'localhost'
PORT = 21567

# 测试未通过
class TSClientProtocol(protocol.Protocol):
    def sendData(self):
        data = input('>')
        if data:
            print('...sending %s' % data)
            print(dir(self.transport))
            # 测试错误：AttributeError: 'NoneType' object has no attribute 'write'
            self.transport.write(bytes(data, 'utf-8'))
        else:
            self.transport.loseConection()

    def connectionMade(self):
        print('connected to server...')
        self.sendData()

    def dataReceived(self, data):
        print(data.decode('utf-8'))
        self.sendData()


class TSClientFactory(protocol.ClientFactory):
    protocol = TSClientProtocol()
    protocol.connectionMade()
    clientConnectionLost = clientConnectionFailed = \
        lambda self, connector, reason: reactor.stop()


reactor.connectTCP(HOST, PORT, TSClientFactory())
reactor.run()

###############################
"""
FTP客户端编程主要流程：
1. 连接到服务器
2. 登录
3. 发出服务请求
4. 退出

伪代码描述如下：
from ftplib import FTP

f = FTP('some.ftp.server')
f.login('anonymous', 'your@email.address')
    :
f.quit()
"""
###############################
# 这个程序用于下载网站中最新版本的文件
# 测试时，无法连接到远程ftp服务器主机
import ftplib
import os
import socket

HOST = 'ftp.mozilla.org'
DIRN = 'pub/mozilla/interviews'
FILE = 'mitchell-baker-interview-oscon-2005.mp3'

def main():
    try:
        ftp = ftplib.FTP(HOST)
    except (socket.error, socket.gaierror) as e:
        print('ERROR: connot reach %s' % HOST)
        return
    print('*** Connected to host %s' % HOST)
    try:
        ftp.login()     # 匿名方式登录
    except ftplib.error_perm:
        print('ERROR: connot login anonymously.')
        ftp.quit()
        return
    print('*** Logged in as anonymous')
    try:
        ftp.cwd(DIRN)   # Change to a directory.
    except ftplib.error_perm:
        print('ERROR: connot CD to "%s"' % DIRN )
        ftp.quit()
        return
    print('***Changed to "%s" folder' % DIRN)
    try:
        local = open(FILE, 'wb')
        ftp.retrbinary('RETR %s ' % FILE, local.write)
    except ftplib.error_perm:
        print('ERROR: connot read file "%s"' % FILE)
        os.unlink(FILE)     # 如果由于某些原因无法保存文件，则移除空的文件来避免弄乱文件系统
    else:
        print('***Download "%s" to CWD' % FILE)
    ftp.quit()


if __name__ == '__main__':
    main()

###############################
#!/usr/bin/python
# -*- coding: UTF-8 -*-
# forked from: https://yq.aliyun.com/articles/444274
from ftplib import FTP
import os
import sys
import time
import socket


class MyFTP:
    """
        ftp自动下载、自动上传脚本，可以递归目录操作
        作者：欧阳鹏
        博客地址：http://blog.csdn.net/ouyang_peng/article/details/79271113
    """

    def __init__(self, host, port=21):
        """ 初始化 FTP 客户端
        参数:
                 host:ip地址

                 port:端口号
        """
        # print("__init__()---> host = %s ,port = %s" % (host, port))

        self.host = host
        self.port = port
        self.ftp = FTP()
        # 重新设置下编码方式
        self.ftp.encoding = 'gbk'
        self.log_file = open("log.txt", "a")
        self.file_list = []

    def login(self, username, password):
        """ 初始化 FTP 客户端
            参数:
                  username: 用户名

                 password: 密码
            """
        try:
            timeout = 60
            socket.setdefaulttimeout(timeout)
            # 0主动模式 1 #被动模式
            self.ftp.set_pasv(True)
            # 打开调试级别2，显示详细信息
            # self.ftp.set_debuglevel(2)

            self.debug_print('开始尝试连接到 %s' % self.host)
            self.ftp.connect(self.host, self.port)
            self.debug_print('成功连接到 %s' % self.host)

            self.debug_print('开始尝试登录到 %s' % self.host)
            self.ftp.login(username, password)
            self.debug_print('成功登录到 %s' % self.host)

            self.debug_print(self.ftp.welcome)
        except Exception as err:
            self.deal_error("FTP 连接或登录失败 ，错误描述为：%s" % err)
            pass

    def is_same_size(self, local_file, remote_file):
        """判断远程文件和本地文件大小是否一致

           参数:
             local_file: 本地文件

             remote_file: 远程文件
        """
        try:
            remote_file_size = self.ftp.size(remote_file)
        except Exception as err:
            # self.debug_print("is_same_size() 错误描述为：%s" % err)
            remote_file_size = -1

        try:
            local_file_size = os.path.getsize(local_file)
        except Exception as err:
            # self.debug_print("is_same_size() 错误描述为：%s" % err)
            local_file_size = -1

        self.debug_print('local_file_size:%d  , remote_file_size:%d' % (local_file_size, remote_file_size))
        if remote_file_size == local_file_size:
            return 1
        else:
            return 0

    def download_file(self, local_file, remote_file):
        """从ftp下载文件
            参数:
                local_file: 本地文件

                remote_file: 远程文件
        """
        self.debug_print("download_file()---> local_path = %s ,remote_path = %s" % (local_file, remote_file))

        if self.is_same_size(local_file, remote_file):
            self.debug_print('%s 文件大小相同，无需下载' % local_file)
            return
        else:
            try:
                self.debug_print('>>>>>>>>>>>>下载文件 %s ... ...' % local_file)
                buf_size = 1024
                file_handler = open(local_file, 'wb')
                self.ftp.retrbinary('RETR %s' % remote_file, file_handler.write, buf_size)
                file_handler.close()
            except Exception as err:
                self.debug_print('下载文件出错，出现异常：%s ' % err)
                return

    def download_file_tree(self, local_path, remote_path):
        """从远程目录下载多个文件到本地目录
                       参数:
                         local_path: 本地路径

                         remote_path: 远程路径
                """
        print("download_file_tree()--->  local_path = %s ,remote_path = %s" % (local_path, remote_path))
        try:
            self.ftp.cwd(remote_path)
        except Exception as err:
            self.debug_print('远程目录%s不存在，继续...' % remote_path + " ,具体错误描述为：%s" % err)
            return

        if not os.path.isdir(local_path):
            self.debug_print('本地目录%s不存在，先创建本地目录' % local_path)
            os.makedirs(local_path)

        self.debug_print('切换至目录: %s' % self.ftp.pwd())

        self.file_list = []
        # 方法回调
        self.ftp.dir(self.get_file_list)

        remote_names = self.file_list
        self.debug_print('远程目录 列表: %s' % remote_names)
        for item in remote_names:
            file_type = item[0]
            file_name = item[1]
            local = os.path.join(local_path, file_name)
            if file_type == 'd':
                print("download_file_tree()---> 下载目录： %s" % file_name)
                self.download_file_tree(local, file_name)
            elif file_type == '-':
                print("download_file()---> 下载文件： %s" % file_name)
                self.download_file(local, file_name)
            self.ftp.cwd("..")
            self.debug_print('返回上层目录 %s' % self.ftp.pwd())
        return True

    def upload_file(self, local_file, remote_file):
        """从本地上传文件到ftp

           参数:
             local_path: 本地文件

             remote_path: 远程文件
        """
        if not os.path.isfile(local_file):
            self.debug_print('%s 不存在' % local_file)
            return

        if self.is_same_size(local_file, remote_file):
            self.debug_print('跳过相等的文件: %s' % local_file)
            return

        buf_size = 1024
        file_handler = open(local_file, 'rb')
        self.ftp.storbinary('STOR %s' % remote_file, file_handler, buf_size)
        file_handler.close()
        self.debug_print('上传: %s' % local_file + "成功!")

    def upload_file_tree(self, local_path, remote_path):
        """从本地上传目录下多个文件到ftp
           参数:

             local_path: 本地路径

             remote_path: 远程路径
        """
        if not os.path.isdir(local_path):
            self.debug_print('本地目录 %s 不存在' % local_path)
            return

        self.ftp.cwd(remote_path)
        self.debug_print('切换至远程目录: %s' % self.ftp.pwd())

        local_name_list = os.listdir(local_path)
        for local_name in local_name_list:
            src = os.path.join(local_path, local_name)
            if os.path.isdir(src):
                try:
                    self.ftp.mkd(local_name)
                except Exception as err:
                    self.debug_print("目录已存在 %s ,具体错误描述为：%s" % (local_name, err))
                self.debug_print("upload_file_tree()---> 上传目录： %s" % local_name)
                self.upload_file_tree(src, local_name)
            else:
                self.debug_print("upload_file_tree()---> 上传文件： %s" % local_name)
                self.upload_file(src, local_name)
        self.ftp.cwd("..")

    def close(self):
        """ 退出ftp
        """
        self.debug_print("close()---> FTP退出")
        self.ftp.quit()
        self.log_file.close()

    def debug_print(self, s):
        """ 打印日志
        """
        self.write_log(s)

    def deal_error(self, e):
        """ 处理错误异常
            参数：
                e：异常
        """
        log_str = '发生错误: %s' % e
        self.write_log(log_str)
        sys.exit()

    def write_log(self, log_str):
        """ 记录日志
            参数：
                log_str：日志
        """
        time_now = time.localtime()
        date_now = time.strftime('%Y-%m-%d', time_now)
        format_log_str = "%s ---> %s \n " % (date_now, log_str)
        print(format_log_str)
        self.log_file.write(format_log_str)

    def get_file_list(self, line):
        """ 获取文件列表
            参数：
                line：
        """
        file_arr = self.get_file_name(line)
        # 去除  . 和  ..
        if file_arr[1] not in ['.', '..']:
            self.file_list.append(file_arr)

    def get_file_name(self, line):
        """ 获取文件名
            参数：
                line：
        """
        pos = line.rfind(':')
        while (line[pos] != ' '):
            pos += 1
        while (line[pos] == ' '):
            pos += 1
        file_arr = [line[0], line[pos:]]
        return file_arr


if __name__ == "__main__":
    my_ftp = MyFTP("172.28.180.117")
    my_ftp.login("ouyangpeng", "ouyangpeng")

    # 下载单个文件
    my_ftp.download_file("G:/ftp_test/XTCLauncher.apk", "/App/AutoUpload/ouyangpeng/I12/Release/XTCLauncher.apk")

    # 下载目录
    # my_ftp.download_file_tree("G:/ftp_test/", "App/AutoUpload/ouyangpeng/I12/")

    # 上传单个文件
    # my_ftp.upload_file("G:/ftp_test/Release/XTCLauncher.apk", "/App/AutoUpload/ouyangpeng/I12/Release/XTCLauncher.apk")

    # 上传目录
    # my_ftp.upload_file_tree("G:/ftp_test/", "/App/AutoUpload/ouyangpeng/I12/")

    my_ftp.close()

###############################
from time import ctime, sleep

# 使用单线程执行循环
def loopOne(sleep_time):
    print('start loop one at: %s' % ctime())
    sleep(sleep_time)
    print('loop one done at: %s' % ctime())

def loopTwo(sleep_time):
    print('start loop two at: %s' % ctime())
    sleep(sleep_time)
    print('loop two done at: %s' % ctime())

def main():
    print('starting at: %s' % ctime())
    loopOne(4)
    loopTwo(6)
    print('all done at: %s' % ctime())


if __name__ == '__main__':
    main()

###############################
from time import ctime, sleep
import _thread

# 使用_thread模块（强烈不建议使用该模块，建议使用高级threading模块）
# 使用不可靠的sleep()进行同步
def loopOne(sleep_time):
    print('start loop one at: %s' % ctime())
    sleep(sleep_time)
    print('loop one done at: %s' % ctime())

def loopTwo(sleep_time):
    print('start loop two at: %s' % ctime())
    sleep(sleep_time)
    print('loop two done at: %s' % ctime())

def main():
    print('starting at: %s' % ctime())
    # start_new_thread()必须包含开始的两个参数，于是即使要执行的函数不需要参数，也需要传递一个空元组
    _thread.start_new_thread(loopOne, (4, ))    # 参数以元祖形式传递
    _thread.start_new_thread(loopTwo, (6, ))
    sleep(10)   # 如不适用该语句，当主线程结束时会强制结束所有子线程
    print('all done at: %s' % ctime())


if __name__ == '__main__':
    main()
###############################
from time import ctime, sleep
import _thread

loops = [4, 2]

# 使用_thread模块，锁进行同步
def loop(loop_number, sleep_time, lock):
    print('start loop %s at: %s' % (loop_number, ctime()))
    sleep(sleep_time)
    print('loop %s done at: %s' % (loop_number, ctime()))
    lock.release()

def main():
    print('starting at: %s' % ctime())
    locks = []
    loop_numbers = range(len(loops))

    # 分配锁
    for i in loop_numbers:
        lock = _thread.allocate_lock()
        lock.acquire()  # 通过acquire()方法取得（每个锁），取得锁效果相当于“上锁”
        locks.append(lock)

    # 启动已上锁线程
    for loop_number in loop_numbers:
        _thread.start_new_thread(loop, (loop_number, loops[loop_number], locks[loop_number]))   # 立即执行

    for i in loop_numbers:
        while locks[i].locked():
            pass

    print('all done at: %s' % ctime())


if __name__ == '__main__':
    main()

###############################
from time import ctime, sleep
import threading

loops = [4, 2]
"""
使用 threading 模块中 Thread 类创建线程

    方法一（推荐）：创建 Thread 的实例，传给它一个函数
"""
def loop(loop_number, sleep_time):
    print('start loop %s at: %s' % (loop_number, ctime()))
    sleep(sleep_time)
    print('loop %s done at: %s' % (loop_number, ctime()))

def main():
    print('starting at: %s' % ctime())
    threads = []
    loop_numbers = range(len(loops))

    for i in loop_numbers:
        thread = threading.Thread(target=loop, args=(i, loops[i]))
        threads.append(thread)

    # 启动已上锁线程
    for i in loop_numbers:
        threads[i].start()  # start threads

    for i in loop_numbers:  
        # join()方法只有在你需要等待线程完成的时候才是有用的，否则无需调用
        threads[i].join()   # 等待线程结束

    print('all done at: %s' % ctime())


if __name__ == '__main__':
    main()

###############################
from time import ctime, sleep
import threading

loops = [4, 2]
"""
使用 threading 模块中 Thread 类创建线程

    方法二（不推荐）：创建 Thread 的实例，传给它一个可调用的类实例
"""
class ThreadFunc(object):
    def __init__(self, func, args, name=''):
        self.func = func
        self.name = name
        self.args = args

    def __call__(self, *args, **kwargs):    # 实现可调用类
        self.func(*self.args)
       
def loop(loop_number, sleep_time):
    print('start loop %s at: %s' % (loop_number, ctime()))
    sleep(sleep_time)
    print('loop %s done at: %s' % (loop_number, ctime()))

def main():
    print('starting at: %s' % ctime())
    threads = []
    loop_numbers = range(len(loops))

    for i in loop_numbers:
        thread = threading.Thread(target=ThreadFunc(loop, (i, loops[i])))
        threads.append(thread)

    # 启动已上锁线程
    for i in loop_numbers:
        threads[i].start()  # start threads

    for i in loop_numbers:
        # join()方法只有在你需要等待线程完成的时候才是有用的，否则无需调用
        threads[i].join()   # 等待线程结束

    print('all done at: %s' % ctime())


if __name__ == '__main__':
    main()

###############################
from time import ctime, sleep
import threading

loops = [4, 2]
"""
使用 threading 模块中 Thread 类创建线程

    方法三（强烈推荐）：派生 Thread 的子类，并创建子类的实例
"""
class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.func = func
        self.name = name
        self.args = args

    def run(self):
        self.func(*self.args)

def loop(loop_number, sleep_time):
    print('start loop %s at: %s' % (loop_number, ctime()))
    sleep(sleep_time)
    print('loop %s done at: %s' % (loop_number, ctime()))

def main():
    print('starting at: %s' % ctime())
    threads = []
    loop_numbers = range(len(loops))

    for i in loop_numbers:
        thread = MyThread(loop, (i, loops[i]))
        threads.append(thread)

    # 启动已上锁线程
    for i in loop_numbers:
        threads[i].start()  # start threads

    for i in loop_numbers:
        # join()方法只有在你需要等待线程完成的时候才是有用的，否则无需调用
        threads[i].join()   # 等待线程结束

    print('all done at: %s' % ctime())


if __name__ == '__main__':
    main()

###############################
import threading
from time import ctime

# 自定义 MyThread 独立模块
class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.name = name
        self.result = None

    def getResult(self):
        return self.result

    def run(self):
        print('starting %s at: %s' % (self.name, ctime()))
        self.result = self.func(*self.args)
        print('%s finished at: %s' % (self.name, ctime()))

###############################
from myThread import MyThread
from time import ctime, sleep

def fib(x):
    sleep(0.005)
    if x < 2:
        return 1
    return fib(x-1) + fib(x-2)

def fac(x):
    sleep(0.1)
    if x < 2:
        return 1
    return x * fac(x-1)

def sum(x):
    if x < 2:
        return 1
    return x + sum(x-1)


funcs = [fib, fac, sum]
n = 12

def main():
    nfuncs = range(len(funcs))

    print('***SINGLE THREAD***')
    for i in nfuncs:
        print('starting %s at: %s' % (funcs[i].__name__, ctime()))
        print(funcs[i](n))
        print('%s finished at: %s' % (funcs[i].__name__, ctime()))

    print('***MULTIPLE THREADS***')
    threads = []
    # 创建线程
    for i in nfuncs:
        thread = MyThread(funcs[i], (n, ), funcs[i].__name__)
        threads.append(thread)
    # 启动线程
    for i in nfuncs:
        threads[i].start()
    #  获取线程执行结果
    for i in nfuncs:
        threads[i].join()
        print(threads[i].getResult())

    print('all done.')


if __name__ == '__main__':
    main()

###############################
from atexit import register
from re import compile
from time import ctime
from urllib.request import urlopen

"""
    本例本身无错，但是可能由于墙的源域导致（HTTP Error 503: Service Unavailable）
    在该简单案例中可以使用正则表达式提取网页中需求信息，但是在网络爬虫阶段则需要更强大的标记解析器，如标准库中的HTMLParser，第三方工具BeautifulSoup、html5lib或者lxml等
"""
REGEX = compile('#([\d,]+) in Books')   # 可使用正则表达式编译bytes类型，如：REGEX = compile(b'#([\d,]+) in Books')，如此则后续read()方法后不需要decode()
AMAZON = 'http://amazon.com/dp/'
ISBNs = {
    '0132678209': 'Core Python Programming, Third Edition',
    '0132356139': 'Python Web Development with Django',
    '0137143419': 'Python Fundamentals',
}

def getRanking(isbn):
    page = urlopen('%s%s' % (AMAZON, isbn))  # or page = uopen('{0}{1}'.format(AMAZON, isbn))
    data = page.read().decode('utf-8')  # 使用read()下载整个网页
    # print(data)   # 打印网页源码
    page.close()    # 关闭文件
    return REGEX.findall(data)[0]   # 如果正则表达式与预期一样精确，应当有且只有一个匹配

def _showRanking(isbn):     # 函数名最前面的单下划线表示这是一个特殊函数，只能被本模块的代码使用，不能被其他使用本文件作为库或者工具模块的应用导入
    print('- %r ranked %s' % (ISBNs[isbn], getRanking(isbn)))

def _main():
    print('At %s on Amazon...' % ctime())
    for isbn in ISBNs:
        _showRanking(isbn)

@register   # 使用装饰器注册退出函数
def _atexit():
    print('all done at: %s' % ctime())


if __name__ == '__main__':
    _main()

###############################
from atexit import register
from re import compile
from time import ctime
from urllib.request import urlopen
from threading import Thread

REGEX = compile('#([\d,]+) in Books')
AMAZON = 'http://amazon.com/dp/'
ISBNs = {
    '0132678209': 'Core Python Programming, Third Edition',
    '0132356139': 'Python Web Development with Django',
    '0137143419': 'Python Fundamentals',
}

def getRanking(isbn):
    page = urlopen('%s%s' % (AMAZON, isbn))  # or page = uopen('{0}{1}'.format(AMAZON, isbn))
    data = page.read().decode('utf-8')  # 使用read()下载整个网页
    # print(data)   # 打印网页源码
    page.close()    # 关闭文件
    return REGEX.findall(data)[0]   # 如果正则表达式与预期一样精确，应当有且只有一个匹配

def _showRanking(isbn):     # 函数名最前面的单下划线表示这是一个特殊函数，只能被本模块的代码使用，不能被其他使用本文件作为库或者工具模块的应用导入
    print('- %r ranked %s' % (ISBNs[isbn], getRanking(isbn)))

def _main():
    print('At %s on Amazon...' % ctime())
    for isbn in ISBNs:
        """
            本例：分配线程并同时启动
            另可：先集体分配线程再集体启动线程
        """
        Thread(target=_showRanking, args=(isbn, )).start()

@register   # 使用装饰器注册退出函数，从而主线程不会在派生线程完成前退出脚本，否则需要使用线程的join()用于阻塞主线程
def _atexit():
    print('all done at: %s' % ctime())


if __name__ == '__main__':
    _main()

###############################
"""
Python 主要移植命令
    1. 2to3 filename.py  # only output diff
    2. 2to3 -w filename.py  # overwrites w/3.x code, and rename 2.x version filename.py to filenme.py.bak
可选步骤：
    1. mv filename2.py filename3.py  # rename the former to the latter
    2. mv filename.py.bak filename2.py

主要移植方法：
    1. 确保源码所有单元测试和集成测试都已通过
    2. 使用2to3（或其他工具）进行所有初步基础修改
    3. 进行进一步善后移植修改，让代码运行起来并通过相同的测试
"""
###############################
from atexit import register
from time import ctime, sleep
from threading import Thread, currentThread
from random import randrange

class CleanOutPutSet(set):
    def __str__(self):
        return ','.join(x for x in self)


loops = (randrange(2, 5) for i in range(randrange(3, 7)))
remaining = CleanOutPutSet()

def loop(sleep_time):
    thread_name = currentThread().name
    remaining.add(thread_name)
    print('[%s] started %s' % (ctime(), thread_name))
    sleep(sleep_time)
    """
        后续由于多线程共同操作共享变量 remaining 且未进行同步而输出异常
        临界区：
            1. I/O
            2. 相同数据结构
    """
    remaining.remove(thread_name)
    print('[%s] completed %s (%d seconds)' % (ctime(), thread_name, sleep_time))
    print('remaining: %s' % remaining or 'None')

def _main():
    for sleep_time in loops:
        Thread(target=loop, args=(sleep_time, )).start()

@register
def _atexit():
    print('all done at: %s' % ctime())


if __name__ == '__main__':
    _main()

###############################
from atexit import register
from time import ctime, sleep
from threading import Thread, currentThread, Lock, enumerate
from random import randrange

"""
    使用 Lock 自带方法访问临界区资源：
        1. lock.acquire()
        2. lock.release() 
"""
class CleanOutPutSet(set):
    def __str__(self):
        return ','.join(x for x in self)


loops = (randrange(2, 5) for i in range(randrange(3, 7)))
remaining = CleanOutPutSet()
lock = Lock()

def loop(sleep_time):
    thread_name = currentThread().name
    lock.acquire()
    remaining.add(thread_name)
    print('[%s] started %s' % (ctime(), thread_name))
    lock.release()
    sleep(sleep_time)
    print(enumerate())  # 以列表形式显示当前正在运行的线程
    lock.acquire()
    remaining.remove(thread_name)
    print('[%s] completed %s (%d seconds)' % (ctime(), thread_name, sleep_time))
    print('remaining: %s' % remaining or 'None')
    lock.release()

def _main():
    for sleep_time in loops:
        Thread(target=loop, args=(sleep_time, )).start()

@register
def _atexit():
    print('all done at: %s' % ctime())


if __name__ == '__main__':
    _main()

###############################
from atexit import register
from time import ctime, sleep
from threading import Thread, currentThread, Lock, enumerate
from random import randrange

"""
    使用上下文管理器访问临界区资源：
        with lock:
            ...
"""
class CleanOutPutSet(set):
    def __str__(self):
        return ','.join(x for x in self)


loops = (randrange(2, 5) for i in range(randrange(3, 7)))
remaining = CleanOutPutSet()
lock = Lock()

def loop(sleep_time):
    thread_name = currentThread().name
    with lock:
        remaining.add(thread_name)
        print('[%s] started %s' % (ctime(), thread_name))
    sleep(sleep_time)
    print(enumerate())  # 以列表形式显示当前正在运行的线程
    with lock:
        remaining.remove(thread_name)
        print('[%s] completed %s (%d seconds)' % (ctime(), thread_name, sleep_time))
        print('remaining: %s' % remaining or 'None')

def _main():
    for sleep_time in loops:
        Thread(target=loop, args=(sleep_time, )).start()

@register
def _atexit():
    print('all done at: %s' % ctime())


if __name__ == '__main__':
    _main()

###############################
from atexit import register
from time import ctime, sleep
from threading import Thread, Lock, BoundedSemaphore
from random import randrange

"""
    使用信号量机制模拟生产者-消费者问题    
"""
lock = Lock()
MAX = 5
candytray = BoundedSemaphore(MAX)  # 限定信号量

def refill():
    lock.acquire()
    print('Refilling candy...')
    if candytray.acquire(False):  # 请求资源，初始请求状态标记 False
        print('Refilled')
    else:
        print('full, skipping')
    lock.release()

def buy():
    lock.acquire()
    print('Buying candy...')
    try:
        candytray.release()  # 释放资源
    except ValueError:
        print('empty, skipping')
    else:
        print('OK')
    lock.release()

def producer(loops):
    for i in range(loops):
        refill()
        sleep(randrange(3))

def consumer(loops):
    for i in range(loops):
        buy()
        sleep(randrange(3))

def _main():
    print('starting at: %s' % ctime())
    nloops = randrange(2, 6)
    print('THE CANDY MACHINE (full with %d bars)!' % MAX)
    Thread(target=producer, args=(nloops,)).start()
    Thread(target=consumer, args=(randrange(nloops, nloops+5), )).start()

@register
def _atexit():
    print('all done at: %s' % ctime())


if __name__ == '__main__':
    _main()

###############################
from atexit import register
from time import ctime, sleep
from threading import Thread, Lock, BoundedSemaphore
from random import randrange

"""
    使用信号量机制模拟生产者-消费者问题    
"""
class MyBoundedSemaphore(BoundedSemaphore):
    def __init__(self, value):
        BoundedSemaphore.__init__(self, value)

    def getSemaphore(self):
        return self._value


lock = Lock()
MAX = 3
candytray = MyBoundedSemaphore(MAX)  # 限定信号量

def refill():
    lock.acquire()
    print('Semaphore = %d, ready to refil candy...' % candytray.getSemaphore())
    if candytray.acquire(False):  # 请求资源，初始请求状态标记 False
        print('Refilled')
    else:
        print('full, skipping')
    lock.release()

def buy():
    lock.acquire()
    print('Semaphore = %d, ready to buy candy...' % candytray.getSemaphore())
    try:
        candytray.release()  # 释放资源
    except ValueError:
        print('empty, skipping')
    else:
        print('OK')
    lock.release()

def producer(loops):
    for i in range(loops):
        refill()
        sleep(randrange(3))

def consumer(loops):
    for i in range(loops):
        buy()
        sleep(randrange(3))

def _main():
    print('starting at: %s' % ctime())
    nloops = randrange(2, 6)
    print('THE CANDY MACHINE (full with %d bars)!' % MAX)
    Thread(target=producer, args=(nloops,)).start()
    Thread(target=consumer, args=(randrange(nloops, nloops+5), )).start()

@register
def _atexit():
    print('all done at: %s' % ctime())


if __name__ == '__main__':
    _main()

###############################
from random import randint
from time import sleep
from queue import Queue
from myThread import MyThread

"""
    使用 Queue/queue 模块模拟生产者-消费者问题
"""
def writeQ(queue):
    print('producing object for Q...')
    queue.put('xxx', 1)
    print('size now is: %d' % queue.qsize())

def readQ(queue):
    queue.get(1)
    print('consumed object from Q...,size now is: %d' % queue.qsize())

def writer(queue, loops):
    for i in range(loops):
        writeQ(queue)
        sleep(randint(1, 3))

def reader(queue, loops):
    for i in range(loops):
        readQ(queue)
        sleep(randint(2, 5))


funcs = [writer, reader]
nfuncs = range(len(funcs))

def _main():
    nloops = randint(2, 5)
    queue = Queue(32)

    threads = []
    for i in nfuncs:
        thread = MyThread(funcs[i], (queue, nloops), funcs[i].__name__)
        threads.append(thread)

    for i in nfuncs:
        threads[i].start()

    for i in nfuncs:
        threads[i].join()

    print('all done...')


if __name__ == '__main__':
    _main()

###############################
"""
使用 concurrent.future 模块中线程池进行高级任务管理主要步骤：

from concurrent.futures import ThreadPoolExecutor
    ...
def _main():
    print('At', ctime(), 'on Amazon...')
    with ThreadPoolExecutor(3) as executor:  #线程池大小为3
        for isbn in ISBNs:
            executor.submit(_showRanking, isbn)
    print('all DONE at:', ctime())
"""
###############################
from concurrent.futures import ThreadPoolExecutor
from re import compile
from time import ctime
from urllib.request import urlopen

"""
    使用 concurrent.future 模块中线程池进行高级任务管理
"""
REGEX = compile('#([\d,]+) in Books')
AMAZON = 'http://amazon.com/dp/'
ISBNs = {
    '0132678209': 'Core Python Programming, Third Edition',
    '0132356139': 'Python Web Development with Django',
    '0137143419': 'Python Fundamentals',
}

def getRanking(isbn):
    with urlopen('{0}{1}'.format(AMAZON, isbn)) as page:
        return REGEX.findall(page.read().decode('utf-8'))[0]

def _main():
    print('At %s on Amazon.' % ctime())
    with ThreadPoolExecutor(3) as executor:
        for isbn, ranking in zip(ISBNs, executor.map(getRanking, ISBNs)):
            print('- %r ranked %s' % (ISBNs[isbn], ranking))
        print('all done at: %s' % ctime())


if __name__ == '__main__':
    _main()

###############################
"""
扩展 python
    完整实现一个扩展主要围绕“封装”相关概念
    实现扩展语言与 python 的无缝连接用到的主要接口代码通常称为样板（boilerplate）代码
    样板代码主要包含以下四个不封：
        1. 在扩展文件中包含 python 头文件   # # include "Python.py"
        2. 为每一个模块函数添加形如 static PyObject* Module_func() 的封装函数  # static PyObject* ModuleName_func()
            此后即可在 python 脚本中导入扩展文件。  # from ModuleName import func
        3. 为每一个模块函数添加一个 PyMethodDef ModuleMethods[]  # 在 python 脚本中声明代理函数，让 Python 解释器知道如何导入并访问这些函数
        4. 添加模块初始化函数 void initModule()

STEP 1:
    纯 C 版本扩展程序（Extest.c）

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    int fac(n)      // 求阶乘模块函数
    {
        if (n < 2):
            return 1;
        else:
            return n * fac(n-1);
    }

    char *reverse(char *s)      // 字符串逆转模块函数
    {
        register char t, *p=s, *q=(s + (len(s) - 1));
        while(p < q)
        {
            t = *p;
            *p++ = *q;
            *q-- = t;
        }
        return s
    }

    BUFSIZE = 1024

    int main()
    {
        char s[BUFSIZE];
        printf("4! == %d\n", fac(4)):
        printf("8! == %d\n", fac(8)):
        printf("12! == %d\n", fac(12)):
        strcpy(s, "admin");
        printf("reversing 'admin', we got '%s'\n", reverse(s));
        return 0;
    }

STEP 2:
    模块函数 fac() 的封装函数 Extest_fac() 如下：

    static PyObject *
    Extest_fac(PyObject *self, PyObject *args){
        int res;    # parse result
        int num;    # arg for fac()
        PyObject * ret_value

        res = PyArg_ParseTuple(args, "i", &num);  // 将 python 元组中的系列参数转换为 C 中的参数
        if (!res)   # TypeError
            return NULL;
        res = fac(num);
        ret_value = (PyObject *)Py_BuildValue("i", res);  // 将 C 数据值转换为 python 返回数据对象
        return ret_value
    }

    简化封装函数 Extest_fac() 如下：

    // 通过使用避免中间变量简化代码
    static PyObject *
    Extest_fac(PyObject *self, PyObject *args){
        if (!PyArg_ParseTuple(args, "i", &num)):
            return NULL;
        return (PyObject *)Py_BuildValue("i", fac(num));
    }

    // 修改 reverse() 使其以元组的形式返回原始字符串和新逆序的字符串
    static PyObject *
    Extest_doppel(PyObject *self, PyObject *args)
    {
        char *origin_str;

        if (!PyArg_ParseTuple(args, "s", &origin_str))
            return NULL;
        // strdup() 用于创建原始字符串的副本，使用“ss”转换字符串将两个字符串放在一个元组中
        // 该部分在 C 语言中会发生内存泄漏，strdup(origin_str)创建新字符串后未回收内存
        return (PyObject *)Py_BuildValue("ss", origin_str, reverse(strdup(origin_str)));
    }

    修正 Extest_doppel() 封装函数如下：

    static PyObject *
    Extest_doppel(PyObject *self, PyObject *args)
    {
        char *origin_str;
        char *dupe_str;
        char re_value;

        dupe_str = strdup(origin_str)

        if (!PyArg_ParseTuple(args, "s", &origin_str))
            return NULL;
        dupe_str = strdup(origin_str);
        // strdup() 用于创建原始字符串的副本，使用“ss”转换字符串将两个字符串放在一个元组中
        re_value = (PyObject *)Py_BuildValue("ss", origin_str, reverse(dupe_str));
        free(dupe_str);    // 释放内存
        return re_value;
    }


    为模块编写 PyMethodDef ModuleMethods[] 数组，让 python 解释器知道如何导入并访问这些函数
    ModuleMethods[] 数组由多个子数组组成，每个子数组含有一个函数的相关信息，母数组以 NULL 数组结尾，表示在此结束

STEP 3:
    为 Extest 模块创建 ExtestMethods[] 数组如下：

    static PyMethodsDef
    ExtestMethods[] = {
        /*
            position_1: 给出在 python 中访问需要使用的名称，
            posiiton_2: 对应的封装函数
            position_3: 常量 METH_VARARGS 表示参数以元组的形式给定
        */

        {"fac", Extest_fac, METH_VARARGS},
        {"doppel", Extest_doppel, METH_VARARGS},
        {NULL, NULL},
    }

STEP 4:
    添加模块初始化函数 void initModule()，当解释器导入该模块时会自动调用该段代码
    如下代码中只调用了 Py_initModule() 函数，这样解释器就可以访问模块函数
    /*
        arg_1: 模块名称
        arg_2: ModuleMethods[] 数组
    */

    void initExtest()
    {
        Py_InitModule("Extest", ExtestMethods);
    }

通过以上 STEP 1、STEP 2、STEP 3、STEP 4 便完成了扩展程序的所有封装任务

为了构建新的 python 封装扩展，需要将其与 python 库一同编译，现在主要使用 distutils 包来构建、安装和发布模块、扩展和软件包

使用 distutils 主要通过以下步骤构建扩展：
    1. 创建 setup.py
    2. 运行 setup.py 来编译并链接代码
    3. 在 Python 中导入模块
    4. 测试函数


STEP 1:
    创建 setup.py
        大部分编译工作由 setup()函数完成，为了构建扩展模块，需要为每个扩展创建一个 Extension 实例

        # arg_1: 扩展的完整名称；arg_2: sources 参数是所有源码文件的列表
        Extension('Extest', sources = ['Extest.c'])

    ...
    【本章后续部分暂停】
"""

###############################