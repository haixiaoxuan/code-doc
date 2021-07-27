## es



memory cache

```properties
GET _cat/nodes?v&h=id,ip,port,r,ramPercent,ramCurrent,heapMax,heapCurrent,fielddataMemory,queryCacheMemory,requestCacheMemory,segmentsMemory


# 报错原因
#indices.breaker.fielddata.limit：此参数设置Fielddata断路器限制大小（公式：预计算内存 + 现有内存 <= 断路器设置内存限制），默认是60%JVM堆内存，当查询尝试加载更多数据到内存时会抛异常（以此来阻止JVM OOM发生）
PUT _cluster/settings
{
  "persistent": {
    "indices": {
      "breaker": {
        "fielddata.limit": "30%"
      }
    }
  }
}

# 清空集群缓存
POST /_cache/clear?pretty

#清空 index 缓存
POST /index/_cache/clear?pretty

#清空 index fielddata 缓存 （推荐，需要等待）
POST /precedent/_cache/clear?fielddata=true
```



force merge

```properties
# 查看某个index的forceMerge情况 
GET /_cat/segments/myindex?v&s=prirep,shard

# 首先查看我们的index(可以使用正则匹配)当前有多少个segment：
GET _cat/segments/myindex?v&h=shard,segment,size,size.memory

# 执行forcemerge：
POST myindex/_forcemerge?max_num_segments=1 

# 查看各个节点forceMerge的线程数：
GET _cat/thread_pool/force_merge?v&s=name
```





查看索引信息

```properties
按照索引大小排序
curl "localhost:9200/_cat/indices?v&s=store.size:desc"
```



re_index

```properties
#查看所有任务
GET _cat/tasks
#查看重建任务
GET _cat/tasks?actions=*reindex
#查看指定id的任务
GET _cat/tasks/{id}
#终止任务
POST _tasks/{id}/_cancel
```

