# AquaSentience：智能图像盲水印保护系统

AquaSentience 是一款专为图像内容安全设计的智能系统，集成了 RAG 检索、多轮对话记忆、工具调用 和 水印嵌入技术。系统主要分为四大模块，为用户提供全面的图像水印保护方案。

## 📌 四大核心模块
1. AI 图像水印专家

基于多轮对话与 RAG 检索，为用户提供水印保护方案咨询。

可以根据用户需求，选择整图或对象级水印策略，并生成总结报告。

2. 整体内容保护（整图水印）

适用于整张图片的全局水印嵌入。

用户上传图片，系统自动嵌入不可见水印，并支持 4 字符水印信息编码（32 位编码）。

3. 局部内容保护（对象级水印）

适合对图像中的特定区域或对象进行水印保护。

用户上传图像及区域掩码，系统进行局部水印嵌入，确保在特定区域嵌入隐蔽且稳健的水印。

4. AI 超级智能体

辅助补充水印专家知识及资源支持多轮对话、记忆持久化、RAG 知识库检索等能力，并且基于 ReAct 模式，能够自主思考并调用工具来完成复杂任务，比如利用网页搜索、资源下载等。




RAG 核心特性及全链路调优：

<img width="1874" height="1562" alt="image" src="https://github.com/user-attachments/assets/793ccad8-9e09-4668-82cc-3cff3274c7dc" />





## 技术栈
基于 Spring Boot 3 +⁠ Spring AI + RAG + Tool Calling + M‌CP + Flask + Pytorch + Cursor前端代码生成
以 Spring AI 开发框架实战为核心，涉及到多种主流 AI 客户端和工具库的运用。

- Java 21 + Spring Boot 3 框架
- ⭐️ Spring AI + LangChain4j
- ⭐️ RAG 知识库
- ⭐️ PGvector 向量数据库
- ⭐ Tool Calling 工具调用 
- ⭐️ MCP 模型上下文协议
- ⭐️ ReAct Agent 智能体构建
- ⭐️ Serverless 计算服务
- ⭐️ AI 大模型开发平台百炼
- ⭐️ Cursor AI 代码生成
- ⭐️ SSE 异步推送
- ⭐️ Pytorch实现深度学习算法
- 第三方接口：如 SearchAPI / Pexels API
- 工具库如：Kryo 高性能序列化 + Jsoup 网页抓取 + iText PDF 生成 + Knife4j 接口文档


RAG 核心特性：

![RAG 核心特性](https://pic.yupi.icu/1/1745224085267-57afea3b-2de9-44a0-8f53-49e338c0e6b9.png)

项目架构设计图：

<img width="640" height="733" alt="image" src="https://github.com/user-attachments/assets/feb11d5a-15f2-479e-8a12-707681a29b78" />







