# AI应用中心

基于Vue3开发的AI应用前端项目，包含AI恋爱大师和AI超级智能体两个聊天应用。

## 功能特性

- 🏠 **主页**：应用选择界面，可以切换不同的AI应用
- 💕 **AI恋爱大师**：专业的恋爱咨询助手，支持实时对话
- 🤖 **AI超级智能体**：强大的AI助手，支持实时对话
- 💬 **聊天室风格**：美观的聊天界面，用户消息在右侧，AI消息在左侧
- 🔄 **SSE实时通信**：使用Server-Sent Events实现实时对话流式传输
- 🎨 **现代化UI**：渐变色彩和流畅动画效果

## 技术栈

- Vue 3.4.0
- Vue Router 4.2.5
- Axios 1.6.0
- Vite 5.0.0

## 项目结构

```
ywyagent_qianduan/
├── index.html              # 入口HTML文件
├── package.json            # 项目配置和依赖
├── vite.config.js          # Vite配置文件
├── src/
│   ├── main.js             # Vue应用入口
│   ├── App.vue             # 根组件
│   ├── router/
│   │   └── index.js        # 路由配置
│   ├── views/
│   │   ├── Home.vue        # 主页
│   │   ├── LoveApp.vue     # AI恋爱大师页面
│   │   └── ManusApp.vue    # AI超级智能体页面
│   ├── components/
│   │   └── ChatRoom.vue    # 聊天室组件
│   └── utils/
│       ├── request.js      # Axios请求配置
│       └── sse.js          # SSE工具函数
└── README.md               # 项目说明文档
```

## 安装和运行

### 1. 安装依赖

```bash
npm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

项目将在 `http://localhost:3000` 启动

### 3. 构建生产版本

```bash
npm run build
```

## 后端接口配置

项目默认连接的后端接口地址为：`http://localhost:8123/api`

如需修改，请编辑 `src/utils/request.js` 文件中的 `baseURL` 配置。

### 接口说明

1. **AI恋爱大师接口**
   - 地址：`GET /ai/love_app/chat/sse`
   - 参数：`message`（消息内容）、`chatId`（聊天室ID）
   - 返回：SSE流式数据

2. **AI超级智能体接口**
   - 地址：`GET /ai/manus/chat`
   - 参数：`message`（消息内容）
   - 返回：SSE流式数据

## 使用说明

1. 启动项目后，访问主页可以看到两个应用卡片
2. 点击"AI恋爱大师"或"AI超级智能体"进入对应的聊天页面
3. 在输入框中输入消息，按回车或点击发送按钮
4. AI回复会实时流式显示在聊天界面中
5. 点击"返回"按钮可以回到主页

## 注意事项

- 确保后端服务已启动并运行在 `http://localhost:8123`
- AI恋爱大师应用会自动生成唯一的聊天室ID，用于区分不同会话
- 聊天消息支持实时流式显示，提供更好的用户体验












