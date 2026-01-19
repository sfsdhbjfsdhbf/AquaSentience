<template>
  <div class="chat-room">
    <div class="chat-header">
      <button class="back-btn" @click="goBack">
        ← 返回
      </button>
      <h2>{{ title }}</h2>
      <div class="chat-id">会话ID: {{ chatId }}</div>
    </div>
    
    <div class="chat-messages" ref="messagesContainer">
      <div
        v-for="(message, index) in messages"
        :key="index"
        :class="['message', message.role]"
      >
        <div class="message-content">
          <div class="message-avatar">
            {{ message.role === 'user' ? '👤' : props.aiAvatar }}
          </div>
          <div class="message-text">
            <div class="message-bubble">
              {{ message.content }}
            </div>
            <!-- PDF下载按钮（支持多个PDF） -->
            <div v-if="message.pdfs && message.pdfs.length > 0" class="pdf-download-container">
              <a 
                v-for="(pdf, idx) in message.pdfs"
                :key="idx"
                :href="getPdfDownloadUrl(pdf.path)" 
                class="pdf-download-btn"
                download
                target="_blank"
                :style="{ marginBottom: idx < message.pdfs.length - 1 ? '8px' : '0' }"
              >
                <span class="pdf-icon">📄</span>
                <span>下载{{ pdf.fileName || 'PDF报告' }}</span>
              </a>
            </div>
            <!-- 资源下载按钮（支持多个资源） -->
            <div v-if="message.resources && message.resources.length > 0" class="resource-download-container">
              <a 
                v-for="(resource, idx) in message.resources"
                :key="idx"
                :href="getResourceDownloadUrl(resource.path)" 
                class="resource-download-btn"
                download
                target="_blank"
                :style="{ marginBottom: idx < message.resources.length - 1 ? '8px' : '0' }"
              >
                <span class="resource-icon">📥</span>
                <span>下载{{ resource.fileName || '资源' }}</span>
              </a>
            </div>
          </div>
        </div>
      </div>
      <div v-if="isLoading" class="message ai">
        <div class="message-content">
          <div class="message-avatar">{{ props.aiAvatar }}</div>
          <div class="message-text">
            <div class="message-bubble loading">
              <span class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- AskHuman 输入框 - 作为消息显示在聊天记录中 -->
      <div v-if="waitingForHumanInput" class="message ask-human-message">
        <div class="message-content">
          <div class="message-avatar">{{ props.aiAvatar }}</div>
          <div class="message-text">
            <div class="ask-human-bubble">
              <div class="ask-human-header">
                <span class="ask-human-icon">⚠️</span>
                <span class="ask-human-title">智能体需要您的帮助</span>
              </div>
              <div class="ask-human-question">
                {{ humanQuestion }}
              </div>
              <div class="ask-human-input-wrapper">
                <input
                  v-model="humanAnswer"
                  type="text"
                  class="ask-human-input"
                  placeholder="请输入您的回答..."
                  @keyup.enter="submitHumanAnswer"
                  ref="humanInputRef"
                />
                <button
                  class="ask-human-submit-btn"
                  @click="submitHumanAnswer"
                  :disabled="!humanAnswer.trim()"
                >
                  提交
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="chat-input-container">
      <div class="chat-input-wrapper">
        <input
          v-model="inputMessage"
          type="text"
          class="chat-input"
          placeholder="输入您的消息..."
          @keyup.enter="sendMessage"
          :disabled="isLoading || waitingForHumanInput"
        />
        <button
          class="send-btn"
          @click="sendMessage"
          :disabled="isLoading || waitingForHumanInput || !inputMessage.trim()"
        >
          发送
        </button>
      </div>
    </div>
    <div class="chat-footer">
      <div class="chat-footer-content">
        <div class="chat-footer-line"></div>
        <p class="chat-footer-text">
          <span class="chat-footer-copyright">© 2026 AI应用中心</span>
          <span class="chat-footer-separator">|</span>
          <span class="chat-footer-author">制作者：<span class="chat-author-name">夜未央</span></span>
          <span class="chat-footer-separator">|</span>
          <a href="mailto:2511209827@qq.com" class="chat-footer-email">
            <span class="chat-email-icon">📧</span>
            2511209827@qq.com
          </a>
        </p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { createSSEConnection, closeSSEConnection } from '../utils/sse'
import { submitHumanAnswer as submitAnswer } from '../utils/askHuman'

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  apiUrl: {
    type: String,
    required: true
  },
  chatId: {
    type: String,
    default: ''
  },
  getParams: {
    type: Function,
    default: () => ({})
  },
  // AI默认头像
  aiAvatar: {
    type: String,
    default: '🤖'
  },
  // 是否为超级智能体应用（需要添加换行）
  isManusApp: {
    type: Boolean,
    default: false
  }
})

const router = useRouter()
const messages = ref([])
const inputMessage = ref('')
const isLoading = ref(false)
const messagesContainer = ref(null)
const waitingForHumanInput = ref(false)
const humanQuestion = ref('')
const humanAnswer = ref('')
const humanInputRef = ref(null)
let eventSource = null
let currentAiMessage = ''
let pendingAiMessageIndex = -1

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

const goBack = () => {
  if (eventSource) {
    closeSSEConnection(eventSource)
  }
  router.push('/')
}

const sendMessage = () => {
  if (!inputMessage.value.trim() || isLoading.value) {
    return
  }

  const userMessage = inputMessage.value.trim()
  inputMessage.value = ''
  
  // 添加用户消息
  messages.value.push({
    role: 'user',
    content: userMessage
  })
  
  scrollToBottom()
  
  // 开始接收AI回复
  isLoading.value = true
  currentAiMessage = ''
  
  // 创建新的AI消息占位
  const aiMessageIndex = messages.value.length
  messages.value.push({
    role: 'ai',
    content: '',
    pdfs: [], // 初始化PDF列表（数组，支持多个PDF）
    resources: [] // 初始化资源列表（数组，支持多个资源）
  })
  
  // 构建请求参数
  const params = props.getParams(userMessage)
  
  // 创建SSE连接
  eventSource = createSSEConnection(
    props.apiUrl,
    params,
    (data) => {
      // 累积接收到的数据
      // 如果是超级智能体应用，每个SSE消息后添加换行
      if (props.isManusApp) {
        currentAiMessage += data + '\n'
      } else {
        currentAiMessage += data
      }
      // 确定要更新的消息索引：如果 pendingAiMessageIndex 有效，使用它；否则使用原来的 aiMessageIndex
      const targetIndex = (pendingAiMessageIndex >= 0 && pendingAiMessageIndex < messages.value.length) 
        ? pendingAiMessageIndex 
        : aiMessageIndex
      // 更新AI消息内容
      if (targetIndex >= 0 && targetIndex < messages.value.length) {
        messages.value[targetIndex].content = currentAiMessage
      }
      scrollToBottom()
    },
    (error) => {
      console.error('SSE错误:', error)
      isLoading.value = false
      waitingForHumanInput.value = false
      if (currentAiMessage) {
        messages.value[aiMessageIndex].content = currentAiMessage
      } else {
        messages.value[aiMessageIndex].content = '抱歉，发生了错误，请重试。'
      }
    },
    () => {
      isLoading.value = false
      waitingForHumanInput.value = false
      if (currentAiMessage) {
        messages.value[aiMessageIndex].content = currentAiMessage
      }
    },
    (question) => {
      // 处理askHuman事件
      handleAskHuman(question, aiMessageIndex)
    },
    (pdfPath) => {
      // 处理PDF生成事件
      handlePdfGenerated(pdfPath, aiMessageIndex)
    },
    (resourcePath, fileName) => {
      // 处理资源下载事件
      handleResourceDownloaded(resourcePath, fileName, aiMessageIndex)
    }
  )
}

// 处理PDF生成事件
const handlePdfGenerated = (pdfPath, aiMessageIndex) => {
  // 确定要更新的消息索引
  const targetIndex = (pendingAiMessageIndex >= 0 && pendingAiMessageIndex < messages.value.length) 
    ? pendingAiMessageIndex 
    : aiMessageIndex
  
  if (targetIndex >= 0 && targetIndex < messages.value.length) {
    // 确保pdfs数组存在
    if (!messages.value[targetIndex].pdfs) {
      messages.value[targetIndex].pdfs = []
    }
    
    // 检查PDF是否已存在（避免重复添加）
    const pdfExists = messages.value[targetIndex].pdfs.some(
      p => p.path === pdfPath
    )
    
    if (!pdfExists) {
      // 添加PDF到数组中（支持多个PDF）
      const fileName = pdfPath.split(/[/\\]/).pop()
      messages.value[targetIndex].pdfs.push({
        path: pdfPath,
        fileName: fileName
      })
      scrollToBottom()
    }
  }
}

// 获取PDF下载URL
const getPdfDownloadUrl = (pdfPath) => {
  // 从路径中提取文件名（例如：C:\Users\...\report.pdf -> report.pdf）
  // 支持Windows路径和Unix路径
  const fileName = pdfPath.split(/[/\\]/).pop()
  // 返回后端下载接口URL，使用path参数传递完整路径
  return `http://localhost:8123/api/ai/pdf/download?path=${encodeURIComponent(pdfPath)}`
}

// 处理资源下载事件
const handleResourceDownloaded = (resourcePath, fileName, aiMessageIndex) => {
  // 确定要更新的消息索引
  const targetIndex = (pendingAiMessageIndex >= 0 && pendingAiMessageIndex < messages.value.length) 
    ? pendingAiMessageIndex 
    : aiMessageIndex
  
  if (targetIndex >= 0 && targetIndex < messages.value.length) {
    // 确保resources数组存在
    if (!messages.value[targetIndex].resources) {
      messages.value[targetIndex].resources = []
    }
    
    // 检查资源是否已存在（避免重复添加）
    const resourceExists = messages.value[targetIndex].resources.some(
      r => r.path === resourcePath
    )
    
    if (!resourceExists) {
      // 添加资源到数组中（支持多个资源）
      messages.value[targetIndex].resources.push({
        path: resourcePath,
        fileName: fileName || resourcePath.split(/[/\\]/).pop()
      })
      scrollToBottom()
    }
  }
}

// 获取资源下载URL
const getResourceDownloadUrl = (resourcePath) => {
  // 返回后端下载接口URL，使用path参数传递完整路径
  return `http://localhost:8123/api/ai/resource/download?path=${encodeURIComponent(resourcePath)}`
}

// 处理askHuman事件
const handleAskHuman = (question, aiMessageIndex) => {
  // 保存当前AI消息索引，用于后续继续更新
  pendingAiMessageIndex = aiMessageIndex
  // 暂停SSE接收（但不关闭连接）
  isLoading.value = false
  waitingForHumanInput.value = true
  humanQuestion.value = question
  humanAnswer.value = ''
  
  // 添加一个系统提示消息
  messages.value.push({
    role: 'system',
    content: `⚠️ 智能体需要您的帮助：${question}`
  })
  
  scrollToBottom()
  
  // 聚焦到输入框
  nextTick(() => {
    if (humanInputRef.value) {
      humanInputRef.value.focus()
    }
  })
}

// 提交用户答案
const submitHumanAnswer = async () => {
  if (!humanAnswer.value.trim()) {
    return
  }
  
  const answer = humanAnswer.value.trim()
  
  // 添加用户回答消息
  messages.value.push({
    role: 'user',
    content: answer
  })
  
  scrollToBottom()
  
  // 关闭askHuman输入框
  waitingForHumanInput.value = false
  isLoading.value = true
  
  try {
    // 提交答案到后端
    // 后端收到答案后，应该继续通过同一个SSE连接发送数据
    await submitAnswer(answer, props.chatId, '')
    
    // 创建新的AI消息占位，显示在用户输入下方
    // 不再更新之前的消息，而是创建新消息
    const newAiMessageIndex = messages.value.length
    messages.value.push({
      role: 'ai',
      content: ''
    })
    currentAiMessage = ''
    
    // 更新 pendingAiMessageIndex，用于接收后续的SSE数据
    pendingAiMessageIndex = newAiMessageIndex
    
    // 注意：后端在收到答案后会继续通过同一个SSE连接发送数据
    // SSE 的 onMessage 回调会更新新创建的 AI 消息
  } catch (error) {
    console.error('提交答案失败:', error)
    isLoading.value = false
    messages.value.push({
      role: 'system',
      content: '提交答案失败，请重试。'
    })
  }
}

// 监听消息变化，自动滚动到底部
watch(messages, () => {
  scrollToBottom()
}, { deep: true })

onMounted(() => {
  scrollToBottom()
})

onUnmounted(() => {
  if (eventSource) {
    closeSSEConnection(eventSource)
  }
})
</script>

<style scoped>
.chat-room {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #f5f5f5;
}

.chat-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 15px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  flex-shrink: 0;
}

@media (max-width: 768px) {
  .chat-header {
    padding: 12px 15px;
  }
  
  .chat-header h2 {
    font-size: 16px;
  }
  
  .chat-id {
    font-size: 10px;
  }
  
  .back-btn {
    padding: 6px 12px;
    font-size: 12px;
  }
}

.back-btn {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  color: white;
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.3s;
}

.back-btn:hover {
  background: rgba(255, 255, 255, 0.3);
}

.chat-header h2 {
  font-size: 20px;
  font-weight: 600;
  flex: 1;
  text-align: center;
}

.chat-id {
  font-size: 12px;
  opacity: 0.9;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

@media (max-width: 768px) {
  .chat-messages {
    padding: 15px;
    gap: 15px;
  }
}

.message {
  display: flex;
  width: 100%;
}

.message.user {
  justify-content: flex-end;
}

.message.ai {
  justify-content: flex-start;
}

.message-content {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  max-width: 70%;
}

@media (max-width: 768px) {
  .message-content {
    max-width: 85%;
    gap: 8px;
  }
  
  .message-avatar {
    width: 32px;
    height: 32px;
    font-size: 16px;
  }
  
  .message-bubble {
    padding: 10px 14px;
    font-size: 14px;
  }
}

.message.user .message-content {
  flex-direction: row-reverse;
}

.message-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
  background: white;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.message-text {
  flex: 1;
}

.message-bubble {
  padding: 12px 16px;
  border-radius: 18px;
  word-wrap: break-word;
  white-space: pre-wrap;
  line-height: 1.5;
}

.message.user .message-bubble {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-bottom-right-radius: 4px;
}

.message.ai .message-bubble {
  background: white;
  color: #333;
  border-bottom-left-radius: 4px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  text-align: left; /* AI回复消息左对齐 */
}

.message.system {
  justify-content: center;
}

.message.system .message-content {
  max-width: 80%;
  background: #fff3cd;
  border: 1px solid #ffc107;
  border-radius: 12px;
  padding: 12px 16px;
  color: #856404;
  font-size: 14px;
}

/* 旧的样式已删除，askHuman 现在作为消息显示 */

.ask-human-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.ask-human-icon {
  font-size: 24px;
}

.ask-human-title {
  font-size: 16px;
  font-weight: 600;
  color: #856404;
}

.ask-human-question {
  background: white;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 12px;
  color: #333;
  font-size: 14px;
  line-height: 1.6;
  border-left: 4px solid #ffc107;
}

.ask-human-input-wrapper {
  display: flex;
  gap: 10px;
}

.ask-human-input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid #ffc107;
  border-radius: 8px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.3s;
}

.ask-human-input:focus {
  border-color: #ff9800;
  box-shadow: 0 0 0 3px rgba(255, 193, 7, 0.1);
}

.ask-human-submit-btn {
  padding: 10px 20px;
  background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.3s;
}

.ask-human-submit-btn:hover:not(:disabled) {
  opacity: 0.9;
}

.ask-human-submit-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.message-bubble.loading {
  background: white;
  padding: 12px 20px;
}

.typing-indicator {
  display: flex;
  gap: 4px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #999;
  animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.7;
  }
  30% {
    transform: translateY(-10px);
    opacity: 1;
  }
}

.chat-input-container {
  background: white;
  border-top: 1px solid #e0e0e0;
  padding: 15px 20px;
  flex-shrink: 0;
}

@media (max-width: 768px) {
  .chat-input-container {
    padding: 12px 15px;
  }
}

.chat-input-wrapper {
  display: flex;
  gap: 10px;
  max-width: 1200px;
  margin: 0 auto;
}

@media (max-width: 768px) {
  .chat-input-wrapper {
    gap: 8px;
  }
  
  .chat-input {
    font-size: 14px;
    padding: 10px 14px;
  }
  
  .send-btn {
    padding: 10px 20px;
    font-size: 14px;
  }
}

.chat-input {
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #e0e0e0;
  border-radius: 24px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.3s;
}

.chat-input:focus {
  border-color: #667eea;
}

.chat-input:disabled {
  background-color: #f5f5f5;
  cursor: not-allowed;
}

.send-btn {
  padding: 12px 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 24px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.3s;
}

.send-btn:hover:not(:disabled) {
  opacity: 0.9;
}

.send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 滚动条样式 */
.chat-messages::-webkit-scrollbar {
  width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.chat-messages::-webkit-scrollbar-thumb {
  background: #888;
  border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
  background: #555;
}

.chat-footer {
  background: white;
  border-top: 1px solid #e0e0e0;
  padding: 20px 20px 25px;
  text-align: center;
  flex-shrink: 0;
  margin-top: auto;
}

.chat-footer-content {
  max-width: 1200px;
  margin: 0 auto;
}

.chat-footer-line {
  width: 80px;
  height: 2px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  margin: 0 auto 15px;
  border-radius: 2px;
}

.chat-footer-text {
  color: #666;
  font-size: 13px;
  line-height: 1.8;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 8px;
  font-weight: 400;
  letter-spacing: 0.3px;
}

.chat-footer-copyright {
  color: #888;
}

.chat-footer-separator {
  color: #ccc;
  margin: 0 4px;
}

.chat-footer-author {
  color: #666;
}

.chat-author-name {
  font-weight: 500;
  color: #333;
}

.chat-footer-email {
  color: #667eea;
  text-decoration: none;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: all 0.3s ease;
  padding: 4px 10px;
  border-radius: 6px;
  background: rgba(102, 126, 234, 0.08);
  font-weight: 500;
}

.chat-footer-email:hover {
  color: #764ba2;
  background: rgba(102, 126, 234, 0.15);
  transform: translateY(-2px);
  text-decoration: none;
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
}

.chat-email-icon {
  font-size: 13px;
}

/* PDF下载按钮样式 */
.pdf-download-container {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.pdf-download-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(231, 76, 60, 0.3);
  width: fit-content;
}

.pdf-download-btn:hover {
  background: linear-gradient(135deg, #c0392b 0%, #a93226 100%);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(231, 76, 60, 0.4);
  text-decoration: none;
  color: white;
}

.pdf-download-btn:active {
  transform: translateY(0);
  box-shadow: 0 2px 6px rgba(231, 76, 60, 0.3);
}

.pdf-icon {
  font-size: 18px;
}

/* 资源下载按钮样式 */
.resource-download-container {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.resource-download-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
  color: white;
  text-decoration: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
  width: fit-content;
}

.resource-download-btn:hover {
  background: linear-gradient(135deg, #2980b9 0%, #21618c 100%);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
  text-decoration: none;
  color: white;
}

.resource-download-btn:active {
  transform: translateY(0);
  box-shadow: 0 2px 6px rgba(52, 152, 219, 0.3);
}

.resource-icon {
  font-size: 18px;
}

@media (max-width: 768px) {
  .pdf-download-btn {
    padding: 8px 14px;
    font-size: 13px;
  }
  
  .pdf-icon {
    font-size: 16px;
  }
  
  .resource-download-btn {
    padding: 8px 14px;
    font-size: 13px;
  }
  
  .resource-icon {
    font-size: 16px;
  }
  
  .chat-footer {
    padding: 15px 15px 20px;
  }
  
  .chat-footer-text {
    font-size: 11px;
    gap: 6px;
    flex-direction: column;
  }
  
  .chat-footer-separator {
    display: none;
  }
  
  .chat-footer-line {
    width: 60px;
    margin-bottom: 12px;
  }
  
  .chat-footer-email {
    padding: 3px 8px;
  }
}
</style>

