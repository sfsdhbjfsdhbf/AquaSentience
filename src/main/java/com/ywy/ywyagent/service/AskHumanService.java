package com.ywy.ywyagent.service;

import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.*;

@Service
public class AskHumanService {

    // 存储等待答案的 Future，key 是 chatId
    private final Map<String, CompletableFuture<String>> pendingAnswers = new ConcurrentHashMap<>();

    // 存储活跃的 SSE 连接，key 是 chatId
    private final Map<String, SseEmitter> activeEmitters = new ConcurrentHashMap<>();

    /**
     * 注册 SSE 连接（在创建 SSE 连接时调用）
     */
    public void registerEmitter(String chatId, SseEmitter emitter) {
        activeEmitters.put(chatId, emitter);

        // SSE 连接关闭时清理
        emitter.onCompletion(() -> {
            activeEmitters.remove(chatId);
            CompletableFuture<String> future = pendingAnswers.remove(chatId);
            if (future != null && !future.isDone()) {
                future.complete("连接已关闭");
            }
        });

        emitter.onTimeout(() -> {
            activeEmitters.remove(chatId);
            CompletableFuture<String> future = pendingAnswers.remove(chatId);
            if (future != null && !future.isDone()) {
                future.complete("连接超时");
            }
        });
    }

    /**
     * 发送 askHuman 事件并等待答案
     */
    public String askHuman(String question, String chatId) throws IOException, InterruptedException, ExecutionException, TimeoutException {
        SseEmitter emitter = activeEmitters.get(chatId);
        if (emitter == null) {
            throw new RuntimeException("未找到对应的SSE连接，chatId: " + chatId);
        }

        // 发送 askHuman 事件给前端
        String jsonData = String.format("{\"question\":\"%s\"}",
                question.replace("\"", "\\\"").replace("\n", "\\n"));

        emitter.send(SseEmitter.event()
                .name("askHuman")
                .data(jsonData));

        // 创建 Future 等待答案
        CompletableFuture<String> answerFuture = new CompletableFuture<>();
        pendingAnswers.put(chatId, answerFuture);

        // 等待答案（最多5分钟）
        try {
            return answerFuture.get(5, TimeUnit.MINUTES);
        } catch (TimeoutException e) {
            pendingAnswers.remove(chatId);
            return "用户未回答（超时）";
        } finally {
            // 确保清理
            pendingAnswers.remove(chatId);
        }
    }

    /**
     * 接收用户答案（由 Controller 调用）
     */
    public void receiveAnswer(String chatId, String answer) {
        CompletableFuture<String> future = pendingAnswers.get(chatId);
        if (future != null && !future.isDone()) {
            future.complete(answer);
        }
    }
}