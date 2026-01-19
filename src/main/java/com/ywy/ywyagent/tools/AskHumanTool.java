package com.ywy.ywyagent.tools;

import com.ywy.ywyagent.context.ChatIdContext;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * AskHumanTool - 向用户询问问题的工具
 *
 * 功能：
 * 1. 当 LLM 需要用户输入时，通过 SSE 发送 askHuman 事件到前端
 * 2. 前端显示输入框，用户输入答案
 * 3. 前端通过 HTTP 接口提交答案
 * 4. 工具等待并返回答案给 LLM
 *
 * 使用方法：
 * 1. 在 Controller 的 SSE 接口中调用：AskHumanTool.registerEmitter(chatId, emitter)
 * 2. 在 Controller 中添加接收答案的接口，调用：AskHumanTool.receiveAnswer(chatId, answer)
 */

@Slf4j
@Component
public class AskHumanTool {

    // 存储等待答案的 Future，key 是 chatId
    private static final Map<String, CompletableFuture<String>> pendingAnswers = new ConcurrentHashMap<>();

    // 存储活跃的 SSE 连接，key 是 chatId
    private static final Map<String, SseEmitter> activeEmitters = new ConcurrentHashMap<>();

    /**
     * LLM 调用的工具，用于向用户询问问题。
     * 工具逻辑：发送SSE事件 → 等待HTTP请求 → 返回给 LLM
     *
     * @param request 包含问题的请求对象
     * @return 包含用户答案的响应对象
     */
    @Tool(name = "askHuman", description = "向人类询问问题，并获得其回答。当需要用户输入信息、确认操作或获取额外上下文时使用此工具。")
    public AskHumanResponse askHuman(AskHumanRequest request) {

        log.info("🧑‍💻 AskHumanTool 被调用，模型想向人类提问：{}", request.getQuestion());

        // 获取 chatId（用于关联 SSE 连接和答案）
        String chatId = getCurrentChatId();

        if (chatId == null || chatId.isEmpty()) {
            log.warn("无法获取 chatId，使用默认值");
            chatId = "default";
        }

        // 1. 获取对应的 SSE 连接
        SseEmitter emitter = activeEmitters.get(chatId);
        if (emitter == null) {
            log.error("未找到对应的SSE连接，chatId: {}，当前活跃连接数: {}", chatId, activeEmitters.size());
            AskHumanResponse response = new AskHumanResponse();
            response.setAnswer("无法连接到前端，请刷新页面重试");
            return response;
        }

        // 2. 通过SSE发送 askHuman 事件给前端
        try {
            // 转义 JSON 特殊字符
            String escapedQuestion = request.getQuestion()
                    .replace("\\", "\\\\")  // 反斜杠
                    .replace("\"", "\\\"")  // 双引号
                    .replace("\n", "\\n")    // 换行
                    .replace("\r", "\\r")    // 回车
                    .replace("\t", "\\t");    // 制表符

            String jsonData = String.format("{\"question\":\"%s\"}", escapedQuestion);

            emitter.send(SseEmitter.event()
                    .name("askHuman")
                    .data(jsonData));

            log.info("✅ 已发送 askHuman 事件到前端，chatId: {}, question: {}", chatId, request.getQuestion());
        } catch (IOException e) {
            log.error("发送 askHuman 事件失败，chatId: {}", chatId, e);
            AskHumanResponse response = new AskHumanResponse();
            response.setAnswer("发送问题到前端失败：" + e.getMessage());
            return response;
        }

        // 3. 创建一个 Future 来等待用户答案
        CompletableFuture<String> answerFuture = new CompletableFuture<>();
        pendingAnswers.put(chatId, answerFuture);

        // 4. 等待用户答案（阻塞等待，最多等待5分钟）
        String userAnswer;
        try {
            log.info("⏳ 等待用户答案，chatId: {}", chatId);
            userAnswer = answerFuture.get(5, TimeUnit.MINUTES);
            log.info("👤 收到人类回答，chatId: {}, answer: {}", chatId, userAnswer);
        } catch (TimeoutException e) {
            log.warn("⏰ 等待用户答案超时，chatId: {}", chatId);
            userAnswer = "用户未回答（超时）";
        } catch (Exception e) {
            log.error("❌ 等待用户答案时发生错误，chatId: {}", chatId, e);
            userAnswer = "获取答案失败：" + e.getMessage();
        } finally {
            // 清理
            pendingAnswers.remove(chatId);
        }

        // 5. 返回答案给 LLM
        AskHumanResponse response = new AskHumanResponse();
        response.setAnswer(userAnswer);

        return response;
    }

    /**
     * 获取当前会话的 chatId
     *
     * 优先级：
     * 1. 从 ThreadLocal 获取（推荐，支持多用户多连接）
     * 2. 如果只有一个活跃连接，使用它的 chatId（适用于单用户或 Reactor 异步场景）
     * 3. 如果无法获取，抛出异常
     *
     * 注意：
     * - 如果使用 Reactor Flux，ThreadLocal 可能在异步线程中丢失
     * - 此时会回退到使用活跃连接的方式（仅当只有一个连接时）
     *
     * @return chatId
     * @throws RuntimeException 如果无法获取 chatId
     */
    private String getCurrentChatId() {
        // 方式1：使用 ThreadLocal（推荐，支持多用户多连接）
        try {
            String chatId = ChatIdContext.get();
            if (chatId != null && !chatId.isEmpty()) {
                log.debug("从 ThreadLocal 获取 chatId: {}", chatId);
                return chatId;
            }
        } catch (Exception e) {
            log.warn("从 ThreadLocal 获取 chatId 失败", e);
        }

        // 方式2：如果只有一个活跃连接，使用它的 chatId
        // 这适用于 Reactor 异步场景，ThreadLocal 可能在不同线程中丢失
        if (activeEmitters.size() == 1) {
            String chatId = activeEmitters.keySet().iterator().next();
            log.warn("ThreadLocal 中无 chatId，从活跃连接获取（单连接场景）: {}", chatId);
            return chatId;
        }

        // 如果无法获取，记录错误并抛出异常
        String errorMsg = String.format(
                "无法获取 chatId！请确保在 Controller 的 SSE 接口中调用了 ChatIdContext.set(chatId)。当前活跃连接数: %d",
                activeEmitters.size()
        );
        log.error(errorMsg);
        throw new RuntimeException(errorMsg);
    }


    // ============================================
    // 静态方法：供 Controller 调用
    // ============================================

    /**
     * 注册 SSE 连接
     *
     * 在 Controller 的 SSE 接口中调用此方法，将 SSE 连接注册到工具中。
     * 这样工具才能通过 SSE 发送 askHuman 事件。
     *
     * 示例：
     * <pre>
     * {@code
     * @GetMapping("/chat/sse")
     * public SseEmitter doChat(String message, String chatId) {
     *     SseEmitter emitter = new SseEmitter(300000L);
     *     AskHumanTool.registerEmitter(chatId, emitter);
     *     // ... 其他逻辑
     *     return emitter;
     * }
     * }
     * </pre>
     *
     * @param chatId 聊天室ID，用于关联连接和答案
     * @param emitter SSE 连接对象
     */
    public static void registerEmitter(String chatId, SseEmitter emitter) {
        // 将 chatId 复制到 final 变量，以便在 lambda 中使用
        final String finalChatId = (chatId == null || chatId.isEmpty()) ? "default" : chatId;

        activeEmitters.put(finalChatId, emitter);
        log.info("📝 注册 SSE 连接，chatId: {}", finalChatId);

        // SSE 连接关闭时清理
        emitter.onCompletion(() -> {
            activeEmitters.remove(finalChatId);
            CompletableFuture<String> future = pendingAnswers.remove(finalChatId);
            if (future != null && !future.isDone()) {
                future.complete("连接已关闭");
            }
            // 清理 ThreadLocal（如果还在当前线程）
            try {
                ChatIdContext.clear();
            } catch (Exception e) {
                // 忽略，可能已经在其他线程
            }
            log.info("🔌 SSE 连接已关闭，chatId: {}", finalChatId);
        });

        emitter.onTimeout(() -> {
            activeEmitters.remove(finalChatId);
            CompletableFuture<String> future = pendingAnswers.remove(finalChatId);
            if (future != null && !future.isDone()) {
                future.complete("连接超时");
            }
            // 清理 ThreadLocal（如果还在当前线程）
            try {
                ChatIdContext.clear();
            } catch (Exception e) {
                // 忽略，可能已经在其他线程
            }
            log.info("⏰ SSE 连接超时，chatId: {}", finalChatId);
        });
    }

    /**
     * 接收用户答案
     *
     * 在 Controller 的接收答案接口中调用此方法，将用户答案传递给等待中的工具。
     *
     * 示例：
     * <pre>
     * {@code
     * @PostMapping("/askHuman/answer")
     * public ResponseEntity<?> receiveAnswer(@RequestBody Map<String, String> request) {
     *     String answer = request.get("answer");
     *     String chatId = request.get("chatId");
     *     AskHumanTool.receiveAnswer(chatId, answer);
     *     return ResponseEntity.ok().build();
     * }
     * }
     * </pre>
     *
     * @param chatId 聊天室ID，用于找到对应的等待
     * @param answer 用户输入的答案
     */
    public static void receiveAnswer(String chatId, String answer) {
        if (chatId == null || chatId.isEmpty()) {
            chatId = "default";
        }

        CompletableFuture<String> future = pendingAnswers.get(chatId);
        if (future != null && !future.isDone()) {
            future.complete(answer);
            log.info("✅ 已接收用户答案，chatId: {}, answer: {}", chatId, answer);
        } else {
            log.warn("⚠️ 未找到对应的等待中的问题，chatId: {}", chatId);
        }
    }

    // -----------------------
    // 请求结构体
    // -----------------------
    @Data
    public static class AskHumanRequest {
        /**
         * 要询问用户的问题
         */
        private String question;
    }

    // -----------------------
    // 响应结构体
    // -----------------------
    @Data
    public static class AskHumanResponse {
        /**
         * 用户的回答
         */
        private String answer;
    }
}
