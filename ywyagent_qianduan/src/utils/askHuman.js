import request from './request'

/**
 * 提交用户答案到后端
 * @param {string} answer - 用户答案
 * @param {string} chatId - 聊天室ID（可选）
 * @param {string} questionId - 问题ID（可选，用于标识是哪个问题）
 * @returns {Promise} 返回Promise
 */
export function submitHumanAnswer(answer, chatId = '', questionId = '') {
  return request.post('/ai/askHuman/answer', {
    answer: answer,
    chatId: chatId,
    questionId: questionId
  })
}















