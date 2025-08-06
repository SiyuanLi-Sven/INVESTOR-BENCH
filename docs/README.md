# INVESTOR-BENCH 项目文档

## 📚 文档索引

- [📖 项目概述](./01-project-overview.md) - 项目介绍、架构和核心概念
- [⚙️ 安装与配置](./02-installation-setup.md) - 环境搭建和配置指南  
- [🚀 快速开始](./03-quick-start.md) - 快速上手指南
- [🔧 配置系统](./04-configuration.md) - 详细配置说明
- [🤖 Agent架构](./05-agent-architecture.md) - FinMemAgent架构详解
- [📈 市场环境](./06-market-environment.md) - MarketEnv组件说明
- [🧠 记忆系统](./07-memory-system.md) - 记忆数据库原理和使用
- [🔌 API接口](./08-api-interfaces.md) - 统一API接口使用指南
- [📊 评估指标](./09-evaluation-metrics.md) - 性能评估和指标分析
- [🛠️ 命令行参考](./10-cli-reference.md) - 完整CLI命令参考
- [🐛 故障排除](./11-troubleshooting.md) - 常见问题和解决方案
- [🔬 开发指南](./12-development-guide.md) - 开发者指南和代码结构

## 🎯 项目简介

INVESTOR-BENCH是一个基于大语言模型(LLM)的智能投资决策评估框架，具有以下核心特性：

### 🌟 核心特性

- **🧠 智能记忆系统**: 多层次记忆机制，支持短期、中期、长期和反思记忆
- **🤖 LLM驱动决策**: 使用大语言模型进行投资决策推理
- **📊 实时市场数据**: 支持股票、加密货币等多种资产类型  
- **⚡ 统一API接口**: OpenAI兼容的统一API调用
- **📈 性能评估**: 全面的投资性能指标分析
- **🔧 灵活配置**: 支持多种模型Provider和参数配置

### 🏗️ 系统架构

```mermaid
graph TB
    subgraph "输入层"
        MD[市场数据]
        NEWS[新闻数据]
        CONFIG[配置文件]
    end
    
    subgraph "核心组件"
        ENV[MarketEnv<br/>市场环境]
        AGENT[FinMemAgent<br/>智能代理]
        MEMORY[Memory System<br/>记忆系统]
    end
    
    subgraph "AI层"
        LLM[LLM Models<br/>大语言模型]
        EMB[Embedding<br/>向量模型]
    end
    
    subgraph "存储层"
        VDB[Qdrant<br/>向量数据库]
        CKPT[Checkpoints<br/>检查点]
    end
    
    subgraph "输出层"
        TRADE[Trading Decisions<br/>交易决策]
        METRICS[Performance Metrics<br/>性能指标]
        LOGS[Execution Logs<br/>执行日志]
    end
    
    MD --> ENV
    NEWS --> ENV
    CONFIG --> AGENT
    
    ENV --> AGENT
    AGENT --> MEMORY
    AGENT --> LLM
    AGENT --> EMB
    
    MEMORY <--> VDB
    AGENT --> CKPT
    
    AGENT --> TRADE
    AGENT --> METRICS
    AGENT --> LOGS
    
    classDef input fill:#e1f5fe
    classDef core fill:#f3e5f5
    classDef ai fill:#fff3e0
    classDef storage fill:#e8f5e8
    classDef output fill:#fce4ec
    
    class MD,NEWS,CONFIG input
    class ENV,AGENT,MEMORY core
    class LLM,EMB ai
    class VDB,CKPT storage
    class TRADE,METRICS,LOGS output
```

### 🎮 工作流程

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Agent
    participant Env as MarketEnv
    participant LLM
    participant Memory
    participant VDB as Qdrant
    
    User->>CLI: 启动warmup
    CLI->>Agent: 初始化Agent
    CLI->>Env: 初始化MarketEnv
    
    loop Warmup阶段
        Env->>Agent: 提供市场观测
        Agent->>Memory: 查询相关记忆
        Memory->>VDB: 检索向量
        VDB-->>Memory: 返回相关记忆
        Memory-->>Agent: 返回记忆上下文
        Agent->>LLM: 生成投资决策
        LLM-->>Agent: 返回决策和推理
        Agent->>Memory: 存储新记忆
        Memory->>VDB: 更新向量数据库
        Agent->>CLI: 保存检查点
    end
    
    User->>CLI: 启动test
    CLI->>Agent: 加载warmup结果
    
    loop Test阶段
        Env->>Agent: 提供市场观测
        Agent->>Memory: 查询记忆
        Agent->>LLM: 生成投资决策
        Agent->>Agent: 执行交易
        Agent->>CLI: 记录性能
    end
    
    User->>CLI: 生成评估报告
    CLI-->>User: 输出性能指标
```

## 🚀 快速导航

### 新用户推荐路径
1. [项目概述](./01-project-overview.md) - 了解基本概念
2. [安装与配置](./02-installation-setup.md) - 搭建环境
3. [快速开始](./03-quick-start.md) - 运行第一个实验
4. [配置系统](./04-configuration.md) - 自定义配置

### 开发者推荐路径  
1. [Agent架构](./05-agent-architecture.md) - 理解核心架构
2. [API接口](./08-api-interfaces.md) - 了解接口设计
3. [开发指南](./12-development-guide.md) - 代码结构和扩展

### 运维推荐路径
1. [命令行参考](./10-cli-reference.md) - 掌握所有命令
2. [故障排除](./11-troubleshooting.md) - 解决常见问题

## 📞 技术支持

如果在使用过程中遇到问题，请按照以下顺序排查：

1. 查阅相关文档章节
2. 检查[故障排除](./11-troubleshooting.md)
3. 查看日志文件获取详细错误信息
4. 运行系统自带的测试脚本

---

📝 **文档版本**: v2.0  
📅 **最后更新**: 2025-01-06  
👨‍💻 **维护者**: 项目开发团队