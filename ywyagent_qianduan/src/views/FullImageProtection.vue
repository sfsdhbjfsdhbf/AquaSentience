<template>
  <div class="full-image-protection">
    <div class="container">
      <div class="header">
        <h1>🛡️ 整图内容保护</h1>
        <p class="subtitle">上传图片并添加水印，或解码已加水印的图片</p>
      </div>

      <!-- 标签页切换 -->
      <div class="tabs">
        <button 
          class="tab-btn" 
          :class="{ 'active': activeTab === 'encode' }"
          @click="activeTab = 'encode'"
        >
          🔒 添加水印
        </button>
        <button 
          class="tab-btn" 
          :class="{ 'active': activeTab === 'decode' }"
          @click="activeTab = 'decode'"
        >
          🔓 解码水印
        </button>
      </div>

      <!-- 添加水印区域 -->
      <div v-show="activeTab === 'encode'" class="content">
        <div class="upload-section">
          <div class="upload-card">
            <h2>上传图片</h2>
            <div class="upload-area" 
                 :class="{ 'dragover': isDragging, 'has-image': originalImage }"
                 @drop="handleDrop"
                 @dragover.prevent="isDragging = true"
                 @dragleave="isDragging = false"
                 @click="triggerFileInput">
              <input 
                ref="fileInput"
                type="file" 
                accept="image/*" 
                @change="handleFileSelect"
                style="display: none"
              />
              
              <div v-if="!originalImage" class="upload-placeholder">
                <div class="upload-icon">📤</div>
                <p class="upload-text">点击或拖拽图片到此处上传</p>
                <p class="upload-hint">支持 JPG、PNG 等格式</p>
              </div>
              
              <div v-else class="preview-original">
                <img :src="originalImage" alt="原始图片" />
                <button class="remove-btn" @click.stop="removeImage">✕</button>
              </div>
            </div>

            <!-- 水印信息输入 -->
            <div class="watermark-input">
              <label for="watermark-msg">水印信息（4位字符）</label>
              <input 
                id="watermark-msg"
                v-model="watermarkMsg" 
                type="text" 
                maxlength="4"
                placeholder="请输入4位字符"
                :disabled="processing"
              />
            </div>

            <!-- 处理按钮 -->
            <button 
              class="process-btn" 
              @click="processImage"
              :disabled="!canProcess || processing"
            >
              <span v-if="processing">处理中...</span>
              <span v-else>添加水印</span>
            </button>
          </div>
        </div>

        <!-- 添加水印结果展示区域 -->
        <div class="result-section">
          <div class="result-card">
            <h2>处理结果</h2>
            <div class="result-content">
              <div class="image-container">
                <img 
                  v-if="watermarkedImage && !imageLoadError"
                  :src="watermarkedImageUrl" 
                  alt="加水印后的图片"
                  @error="handleImageError"
                  @load="handleImageLoad"
                />
                <div v-if="watermarkedImage && imageLoadError" class="error-message">
                  <p>⚠️ 图片加载失败 (404)</p>
                  <p class="error-url">URL: {{ watermarkedImageUrl }}</p>
                  <p class="error-hint">请检查后端是否正确配置了静态文件服务</p>
                  <p class="error-hint">确保 /static/wam/ 路径可以访问</p>
                </div>
                <div v-if="!watermarkedImage" class="placeholder-message">
                  <p>等待处理结果...</p>
                </div>
              </div>
              <div class="result-info" v-if="watermarkedImage">
                <div class="info-item">
                  <span class="info-label">水印信息：</span>
                  <span class="info-value">{{ resultData.msg_text }}</span>
                </div>
                <div class="info-item">
                  <span class="info-label">32位编码：</span>
                  <span class="info-value code">{{ resultData.bits_32 }}</span>
                </div>
              </div>
              <div class="download-row persistent-row" v-if="watermarkedImage && (downloadUrl || watermarkedImageUrl)">
                <button class="download-btn persistent" @click="downloadWatermarked">
                  <span>⬇️</span> 下载水印图片
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 解码水印区域 -->
      <div v-show="activeTab === 'decode'" class="content">
        <div class="upload-section">
          <div class="upload-card">
            <h2>上传加水印的图片</h2>
            <div class="upload-area" 
                 :class="{ 'dragover': isDecodeDragging, 'has-image': decodeImage }"
                 @drop="handleDecodeDrop"
                 @dragover.prevent="isDecodeDragging = true"
                 @dragleave="isDecodeDragging = false"
                 @click="triggerDecodeFileInput">
              <input 
                ref="decodeFileInput"
                type="file" 
                accept="image/*" 
                @change="handleDecodeFileSelect"
                style="display: none"
              />
              
              <div v-if="!decodeImage" class="upload-placeholder">
                <div class="upload-icon">📤</div>
                <p class="upload-text">点击或拖拽加水印的图片到此处</p>
                <p class="upload-hint">支持 JPG、PNG 等格式</p>
              </div>
              
              <div v-else class="preview-original">
                <img :src="decodeImage" alt="加水印的图片" />
                <button class="remove-btn" @click.stop="removeDecodeImage">✕</button>
              </div>
            </div>

            <!-- 解码按钮 -->
            <button 
              class="process-btn" 
              @click="startDecode"
              :disabled="!decodeImage || decoding"
            >
              <span v-if="decoding">解码中...</span>
              <span v-else>开始解码</span>
            </button>
          </div>
        </div>

        <!-- 解码结果展示区域 -->
        <div class="result-section">
          <div class="result-card">
            <h2>解码结果</h2>
            <div class="result-content">
              <div class="image-container">
                <img 
                  v-if="decodeResult && decodeResult.pred_mask_url"
                  :src="decodeMaskUrl" 
                  alt="预测的掩码"
                />
                <div v-if="!decodeResult" class="placeholder-message">
                  <p>等待处理结果...</p>
                </div>
              </div>
              <div class="result-info" v-if="decodeResult">
                <div class="info-item">
                  <span class="info-label">解码信息：</span>
                  <span class="info-value highlight">{{ decodeResult.decoded_msg_text }}</span>
                </div>
                <div class="info-item">
                  <span class="info-label">32位编码：</span>
                  <span class="info-value code">{{ decodeResult.decoded_bits_32 }}</span>
                </div>
              </div>
              <div class="download-row persistent-row" v-if="decodeResult && decodeMaskDownloadUrl">
                <button class="download-btn persistent" @click="downloadDecodeMask">
                  <span>⬇️</span> 下载掩码
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import axios from 'axios'

// 标签页状态
const activeTab = ref('encode')

// 添加水印相关
const fileInput = ref(null)
const originalImage = ref('')
const watermarkMsg = ref('')
const watermarkedImage = ref('')
const watermarkedImageUrl = ref('')
const downloadUrl = ref('')
const resultData = ref({})
const processing = ref(false)
const isDragging = ref(false)
const selectedFile = ref(null)
const imageLoadError = ref(false)

// 解码相关
const decodeFileInput = ref(null)
const decodeImage = ref('')
const decodeResult = ref(null)
const decodeMaskUrl = ref('')
const decodeMaskDownloadUrl = ref('')
const decoding = ref(false)
const isDecodeDragging = ref(false)
const selectedDecodeFile = ref(null)

const canProcess = computed(() => {
  return originalImage.value && watermarkMsg.value.length === 4 && !processing.value
})

const triggerFileInput = () => {
  fileInput.value?.click()
}

const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    loadImage(file)
  }
}

const handleDrop = (event) => {
  event.preventDefault()
  isDragging.value = false
  const file = event.dataTransfer.files[0]
  if (file && file.type.startsWith('image/')) {
    loadImage(file)
  }
}

const loadImage = (file) => {
  selectedFile.value = file
  const reader = new FileReader()
  reader.onload = (e) => {
    originalImage.value = e.target.result
    // 重置结果
    watermarkedImage.value = ''
    watermarkedImageUrl.value = ''
    downloadUrl.value = ''
    resultData.value = {}
    imageLoadError.value = false
  }
  reader.readAsDataURL(file)
}

const removeImage = () => {
  originalImage.value = ''
  selectedFile.value = null
  watermarkedImage.value = ''
  watermarkedImageUrl.value = ''
  downloadUrl.value = ''
  resultData.value = {}
  watermarkMsg.value = ''
  imageLoadError.value = false
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const handleImageError = (event) => {
  console.error('图片加载失败:', event)
  imageLoadError.value = true
}

const handleImageLoad = () => {
  console.log('图片加载成功')
  imageLoadError.value = false
}

const processImage = async () => {
  if (!canProcess.value) return

  processing.value = true
  
  try {
    const formData = new FormData()
    formData.append('image', selectedFile.value)
    formData.append('msg', watermarkMsg.value)

    const response = await axios.post('http://localhost:5000/encode_full', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 60000
    })

    if (response.data) {
      resultData.value = response.data
      
      const baseUrl = 'http://localhost:5000'
      
      const buildUrl = (url) => {
        if (!url) return ''
        if (url.startsWith('http://') || url.startsWith('https://')) {
          return url
        }
        const path = url.startsWith('/') ? url : '/' + url
        return baseUrl + path
      }
      
      watermarkedImageUrl.value = buildUrl(response.data.watermarked_image_url)
      downloadUrl.value = buildUrl(response.data.download_url)
      
      watermarkedImage.value = watermarkedImageUrl.value
      imageLoadError.value = false
      
      console.log('后端返回数据:', response.data)
      console.log('构建的图片URL:', watermarkedImageUrl.value)
      console.log('构建的下载URL:', downloadUrl.value)
    }
  } catch (error) {
    console.error('处理图片失败:', error)
    alert(error.response?.data?.error || '处理图片失败，请重试')
  } finally {
    processing.value = false
  }
}

// 下载加水印图片
const downloadWatermarked = async () => {
  const targetUrl = downloadUrl.value || watermarkedImageUrl.value
  if (!targetUrl) {
    alert('暂无可下载的图片，请先添加水印')
    return
  }
  try {
    const response = await axios.get(targetUrl, { responseType: 'blob' })
    const blob = new Blob([response.data])
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    const filename = targetUrl.split('/').pop() || 'watermarked.png'
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  } catch (error) {
    console.error('下载图片失败:', error)
    alert('下载图片失败，请检查静态资源访问或重试')
  }
}

// 解码相关函数
const triggerDecodeFileInput = () => {
  decodeFileInput.value?.click()
}

const handleDecodeFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    loadDecodeImage(file)
  }
}

const handleDecodeDrop = (event) => {
  event.preventDefault()
  isDecodeDragging.value = false
  const file = event.dataTransfer.files[0]
  if (file && file.type.startsWith('image/')) {
    loadDecodeImage(file)
  }
}

const loadDecodeImage = (file) => {
  selectedDecodeFile.value = file
  const reader = new FileReader()
  reader.onload = (e) => {
    decodeImage.value = e.target.result
    decodeResult.value = null
    decodeMaskUrl.value = ''
    decodeMaskDownloadUrl.value = ''
  }
  reader.readAsDataURL(file)
}

const removeDecodeImage = () => {
  decodeImage.value = ''
  selectedDecodeFile.value = null
  decodeResult.value = null
  decodeMaskUrl.value = ''
  decodeMaskDownloadUrl.value = ''
  if (decodeFileInput.value) {
    decodeFileInput.value.value = ''
  }
}

const startDecode = async () => {
  if (!selectedDecodeFile.value || decoding.value) return

  decoding.value = true
  
  try {
    const formData = new FormData()
    formData.append('image', selectedDecodeFile.value)

    const response = await axios.post('http://localhost:5000/decode', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 60000
    })

    if (response.data) {
      decodeResult.value = response.data
      
      const baseUrl = 'http://localhost:5000'
      
      const buildUrl = (url) => {
        if (!url) return ''
        if (url.startsWith('http://') || url.startsWith('https://')) {
          return url
        }
        const path = url.startsWith('/') ? url : '/' + url
        return baseUrl + path
      }
      
      if (response.data.pred_mask_url) {
        decodeMaskUrl.value = buildUrl(response.data.pred_mask_url)
        if (response.data.download_url) {
          decodeMaskDownloadUrl.value = buildUrl(response.data.download_url)
        } else {
          const urlPath = response.data.pred_mask_url.startsWith('http') 
            ? new URL(response.data.pred_mask_url).pathname 
            : response.data.pred_mask_url
          const filename = urlPath.split('/').pop()
          decodeMaskDownloadUrl.value = `${baseUrl}/download?file=${filename}`
        }
        
        console.log('解码结果:', response.data)
        console.log('构建的掩码URL:', decodeMaskUrl.value)
        console.log('构建的下载URL:', decodeMaskDownloadUrl.value)
      }
    }
  } catch (error) {
    console.error('解码图片失败:', error)
    alert(error.response?.data?.error || '解码图片失败，请重试')
  } finally {
    decoding.value = false
  }
}

// 下载解码掩码
const downloadDecodeMask = async () => {
  if (!decodeMaskDownloadUrl.value) {
    alert('暂无可下载的掩码')
    return
  }
  try {
    const response = await axios.get(decodeMaskDownloadUrl.value, { responseType: 'blob' })
    const blob = new Blob([response.data])
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    const filename = decodeMaskDownloadUrl.value.split('/').pop().split('?')[0] || 'mask.png'
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  } catch (error) {
    console.error('下载掩码失败:', error)
    alert('下载掩码失败，请检查静态资源访问或重试')
  }
}
</script>

<style scoped>
.full-image-protection {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 40px 20px;
}

.container {
  max-width: 1500px;
  margin: 0 auto;
}

.header {
  text-align: center;
  margin-bottom: 40px;
  color: white;
}

.header h1 {
  font-size: 42px;
  font-weight: 700;
  margin-bottom: 10px;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle {
  font-size: 18px;
  opacity: 0.9;
}

.tabs {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-bottom: 30px;
}

.tab-btn {
  padding: 12px 30px;
  background: rgba(255, 255, 255, 0.2);
  color: white;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 25px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.tab-btn:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: translateY(-2px);
}

.tab-btn.active {
  background: white;
  color: #667eea;
  border-color: white;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}


.content {
  display: grid;
  grid-template-columns: 1fr;
  gap: 40px;
}

@media (min-width: 1100px) {
  .content {
    grid-template-columns: 1fr 1fr;
    gap: 40px;
    align-items: start;
  }
}

.upload-section,
.result-section {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
}

.upload-card,
.result-card {
  background: white;
  border-radius: 20px;
  padding: 36px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  width: 100%;
  display: flex;
  flex-direction: column;
  flex: 1;
  box-sizing: border-box;
}

.upload-card h2,
.result-card h2 {
  font-size: 24px;
  color: #333;
  margin-bottom: 20px;
  text-align: center;
}

.upload-area {
  border: 3px dashed #667eea;
  border-radius: 15px;
  padding: 50px 30px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: #f8f9ff;
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  margin-bottom: 24px;
  width: 100%;
}

.upload-area:hover {
  border-color: #764ba2;
  background: #f0f2ff;
}

.upload-area.dragover {
  border-color: #764ba2;
  background: #e8ebff;
  transform: scale(1.02);
}

.upload-area.has-image {
  padding: 0;
  border: none;
  background: transparent;
  min-height: auto;
}

.upload-placeholder {
  width: 100%;
}

.upload-icon {
  font-size: 64px;
  margin-bottom: 15px;
}

.upload-text {
  font-size: 18px;
  color: #667eea;
  font-weight: 600;
  margin-bottom: 8px;
}

.upload-hint {
  font-size: 14px;
  color: #999;
}

.preview-original {
  position: relative;
  width: 100%;
  border-radius: 10px;
  overflow: hidden;
}

.preview-original img {
  width: 100%;
  height: auto;
  display: block;
  max-height: 400px;
  object-fit: contain;
}

.remove-btn {
  position: absolute;
  top: 10px;
  right: 10px;
  background: rgba(255, 0, 0, 0.8);
  color: white;
  border: none;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  font-size: 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
}

.remove-btn:hover {
  background: rgba(255, 0, 0, 1);
  transform: scale(1.1);
}

.watermark-input {
  margin-bottom: 20px;
}

.watermark-input label {
  display: block;
  font-size: 16px;
  color: #333;
  margin-bottom: 8px;
  font-weight: 600;
}

.watermark-input input {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #e0e0e0;
  border-radius: 10px;
  font-size: 16px;
  transition: all 0.3s ease;
}

.watermark-input input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.watermark-input input:disabled {
  background: #f5f5f5;
  cursor: not-allowed;
}

.process-btn {
  width: 100%;
  padding: 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 10px;
  font-size: 18px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.process-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
}

.process-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.result-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.image-container {
  position: relative;
  border-radius: 10px;
  overflow: hidden;
  background: #f5f5f5;
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  margin-bottom: 20px;
}

.image-container img {
  width: 100%;
  height: auto;
  display: block;
  max-height: 380px;
  object-fit: contain;
}

.placeholder-message {
  padding: 60px 20px;
  text-align: center;
  color: #999;
  font-size: 16px;
}

.error-message {
  padding: 40px 20px;
  text-align: center;
  color: #e74c3c;
  background: #fff5f5;
  border: 2px dashed #e74c3c;
  border-radius: 10px;
}

.error-message p {
  margin: 10px 0;
}

.error-url {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #666;
  word-break: break-all;
  background: white;
  padding: 8px;
  border-radius: 4px;
  margin: 10px 0;
}

.error-hint {
  font-size: 14px;
  color: #999;
  margin-top: 5px;
}

.result-info {
  background: #f8f9ff;
  padding: 20px;
  border-radius: 10px;
}

.info-item {
  margin-bottom: 12px;
  display: flex;
  align-items: flex-start;
}

.info-item:last-child {
  margin-bottom: 0;
}

.info-label {
  font-weight: 600;
  color: #333;
  min-width: 100px;
}

.info-value {
  color: #666;
  word-break: break-all;
}

.info-value.code {
  font-family: 'Courier New', monospace;
  font-size: 14px;
  background: white;
  padding: 4px 8px;
  border-radius: 4px;
}

.info-value.highlight {
  font-size: 20px;
  font-weight: 700;
  color: #667eea;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
  padding: 8px 16px;
  border-radius: 8px;
}

.download-row {
  display: flex;
  justify-content: center;
  margin-top: 10px;
}

.download-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 12px 24px;
  border-radius: 8px;
  text-decoration: none;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  transition: all 0.3s ease;
  border: none;
  cursor: pointer;
  font-size: 16px;
}

.download-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}

.download-btn.persistent {
  width: 100%;
  justify-content: center;
}

@media (max-width: 768px) {
  .full-image-protection {
    padding: 20px 15px;
  }

  .header h1 {
    font-size: 32px;
  }

  .subtitle {
    font-size: 16px;
  }

  .tabs {
    margin-bottom: 20px;
  }

  .tab-btn {
    padding: 10px 20px;
    font-size: 14px;
  }

  .upload-card,
  .result-card {
    padding: 20px;
  }

  .upload-area {
    padding: 30px 20px;
    min-height: 250px;
  }

  .upload-icon {
    font-size: 48px;
  }

  .upload-text {
    font-size: 16px;
  }

  .info-item {
    flex-direction: column;
    gap: 4px;
  }

  .info-label {
    min-width: auto;
  }
}
</style>
