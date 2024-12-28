# llms4subjects

## Embeddings

模型：Snowflake/snowflake-arctic-embed-l-v2.0，突出英文和德文的能力

## Match方式：

1. 对标题和摘要进行Embedding，查找出相似的文献，进而得出文献对应的GND Code，将
   GND Codes按照如下规则进行合并排序：

  - 排在前面的code具有更高的权重
  - 在不同样本中重复出现的code具有更高的权重
  - 在标题中出现的name，对应的code具有更高的权重
  - 在摘要中出现的name，对应的code具有更高的权重


2. 对标题和摘要的Embedding，直接在主题库中搜索，返回语义最相似的topk个结果。并
   按照如下如下规则排序：
   - 在标题中出现的name，对应的code具有更高的权重
   - 在摘要中出现的name，对应的code具有更高的权重 


## 数据特点

每个GND Code都拥有一个标准的Name和唯一的code，同时会有多个候选名称。例如：

```json
{
    "Code": "gnd:4056795-3",
    "Classification Number": "31.3a",
    "Classification Name": "Architektur",
    "Name": "Sta\u0308dtebau",
    "Alternate Name": [
        "Stadtarchitektur",
        "Stadtbaukunst",
        "Urbanistik"
    ],
    "Related Subjects": [
        "Stadtgestaltung"
    ],
    "Source": "B 1986"
}
```

需要注意的是，可选名称可能同时是某一个subjec的规范名称，例如，``Städtebau''既是
``gnd:4056795-3''的规范名称，也是以下``gnd:7504235-6''的可选名称。

``` json
{
    "Code": "gnd:7504235-6",
    "Classification Number": "10.6a",
    "Classification Name": "Telekommunikation und Verkehr",
    "Name": "Stadtverkehrsplanung",
    "Alternate Name": [
        "Sta\u0308dtebau",
        "Stadt",
        "Stadtverkehr"
    ],
    "Related Subjects": [],
    "Source": "RSWK \u00a7 324.3 (3.Aufl.)"
}
```
为解决上述问题，我们采用如下的方式对主题构建Embedding：name+alternate name+classificatiion name, 例如，上面的例子，构建的文本为：

``` text
Subjects: Telekommunikation und Verkehr; Städtebau; Stadt;Stadtverkehr
Classification Name: Telekommunikation und Verkehr
```
