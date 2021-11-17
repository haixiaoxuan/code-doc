import asyncio


"""
    asyncio.create_task(func()) 
    此代码执行的时候必须要求 事件循环 已经被创建
"""

# ############################################ 第一种方式 #######################

async def func():
    await asyncio.sleep(2)
    return


async def main():
    print("start")
    task_list = [
        asyncio.create_task(func()),
        asyncio.create_task(func())
    ]

    done, pending = await asyncio.wait(task_list, timeout=None)
    print(done)


asyncio.run(main())


# ############################################ 第二种方式 #######################

task_list = [
    func(),
    func()
]
done, pending = asyncio.run(asyncio.wait(task_list))
print(done)


"""
    Future 示例
"""

async def set_after(fut):
    await asyncio.sleep(2)
    fut.set_result("hello")

async def main():
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await loop.create_task(set_after(fut))

    data = await fut
    print(data)

asyncio.run(main())


"""
    concurrent.futures.Future 示例
    如果使用第三方模块，不支持基于协程的异步，就使用这种方式
"""
def func1():
    import time
    time.sleep(1)
    return "hello"

async def main():
    loop = asyncio.get_event_loop()
    # step1. 调用ThreadPoolExecutor的submit方法，申请一个线程去执行func1，并返回 concurrent.futures.Future 对象
    # step2. 调用 asyncio.wrap_future 将 concurrent.futures.Future 封装为 asyncio.Future 对象
    fut = loop.run_in_executor(None, func1)
    result = await fut
    print(result)

asyncio.run(main())
