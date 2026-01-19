<template>
  <ChatRoom
    title="AI超级智能体"
    :api-url="apiUrl"
    :chat-id="chatId"
    :get-params="getParams"
    ai-avatar="🤖"
    :is-manus-app="true"
  />
</template>

<script setup>
import { ref, onMounted } from 'vue'
import ChatRoom from '../components/ChatRoom.vue'

const apiUrl = 'http://localhost:8123/api/ai/manus/chat'
const chatId = ref('')

// 生成唯一的聊天室ID
const generateChatId = () => {
  return `manus_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

// 获取请求参数
const getParams = (message) => {
  return {
    message: message,
    chatId: chatId.value
  }
}

onMounted(() => {
  chatId.value = generateChatId()
})
</script>

