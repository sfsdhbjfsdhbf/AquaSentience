/**
 * 创建SSE连接
 * @param {string} url - SSE接口地址
 * @param {Object} params - 请求参数
 * @param {Function} onMessage - 消息回调函数
 * @param {Function} onError - 错误回调函数
 * @param {Function} onComplete - 完成回调函数
 * @param {Function} onAskHuman - askHuman事件回调函数，接收question参数
 * @param {Function} onPdfGenerated - pdfGenerated事件回调函数，接收pdfPath参数
 * @param {Function} onResourceDownloaded - resourceDownloaded事件回调函数，接收resourcePath和fileName参数
 * @returns {EventSource} EventSource实例
 */
export function createSSEConnection(url, params, onMessage, onError, onComplete, onAskHuman, onPdfGenerated, onResourceDownloaded) {
  // 构建查询字符串
  const queryString = new URLSearchParams(params).toString()
  const fullUrl = `${url}?${queryString}`

  const eventSource = new EventSource(fullUrl)

  eventSource.onmessage = (event) => {
    if (onMessage) {
      onMessage(event.data)
    }
  }

  // 监听askHuman事件
  eventSource.addEventListener('askHuman', (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.question && onAskHuman) {
        onAskHuman(data.question)
      }
    } catch (e) {
      console.error('解析askHuman事件数据失败:', e)
    }
  })

  // 监听pdfGenerated事件
  eventSource.addEventListener('pdfGenerated', (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.pdfPath && onPdfGenerated) {
        onPdfGenerated(data.pdfPath)
      } else if (typeof event.data === 'string' && onPdfGenerated) {
        // 如果后端直接发送字符串路径，也支持
        onPdfGenerated(event.data)
      }
    } catch (e) {
      // 如果解析失败，尝试直接使用原始数据
      if (event.data && onPdfGenerated) {
        onPdfGenerated(event.data)
      } else {
        console.error('解析pdfGenerated事件数据失败:', e)
      }
    }
  })

  // 监听resourceDownloaded事件
  eventSource.addEventListener('resourceDownloaded', (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.resourcePath && onResourceDownloaded) {
        onResourceDownloaded(data.resourcePath, data.fileName)
      }
    } catch (e) {
      console.error('解析resourceDownloaded事件数据失败:', e)
    }
  })

  eventSource.onerror = (error) => {
    console.error('SSE连接错误:', error)
    if (onError) {
      onError(error)
    }
    eventSource.close()
  }

  // 监听连接关闭
  eventSource.addEventListener('close', () => {
    if (onComplete) {
      onComplete()
    }
    eventSource.close()
  })

  return eventSource
}

/**
 * 关闭SSE连接
 * @param {EventSource} eventSource - EventSource实例
 */
export function closeSSEConnection(eventSource) {
  if (eventSource) {
    eventSource.close()
  }
}

