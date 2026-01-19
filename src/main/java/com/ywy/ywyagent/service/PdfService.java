package com.ywy.ywyagent.service;

import com.ywy.ywyagent.context.ChatIdContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * PDF服务 - 管理SseEmitter并发送PDF生成事件
 */
@Service
public class PdfService {

    private static final Logger log = LoggerFactory.getLogger(PdfService.class);

    // 存储活跃的SSE连接，key为chatId
    private final Map<String, SseEmitter> activeEmitters = new ConcurrentHashMap<>();

    /**
     * 注册SSE连接
     * 在Controller的SSE接口中调用此方法
     */
    public void registerEmitter(String chatId, SseEmitter emitter) {
        activeEmitters.put(chatId, emitter);

        // 连接完成时清理
        emitter.onCompletion(() -> {
            log.info("SSE连接完成，清理chatId: {}", chatId);
            activeEmitters.remove(chatId);
        });

        // 连接超时时清理
        emitter.onTimeout(() -> {
            log.info("SSE连接超时，清理chatId: {}", chatId);
            activeEmitters.remove(chatId);
        });

        // 连接错误时清理
        emitter.onError((ex) -> {
            log.error("SSE连接错误，清理chatId: {}", chatId, ex);
            activeEmitters.remove(chatId);
        });
    }

    /**
     * 发送PDF生成事件到前端
     *
     * @param chatId 聊天ID
     * @param pdfPath PDF文件的完整路径
     */
    public void sendPdfGeneratedEvent(String chatId, String pdfPath) {
        SseEmitter emitter = activeEmitters.get(chatId);
        if (emitter == null) {
            log.warn("未找到对应的SSE连接，无法发送PDF生成事件，chatId: {}", chatId);
            return;
        }

        try {
            // 转义路径中的反斜杠和引号，用于JSON格式
            String escapedPath = pdfPath.replace("\\", "\\\\").replace("\"", "\\\"");
            String jsonData = String.format("{\"pdfPath\":\"%s\"}", escapedPath);

            emitter.send(SseEmitter.event()
                    .name("pdfGenerated")
                    .data(jsonData));

            log.info("✅ 已发送PDF生成事件到前端，chatId: {}, pdfPath: {}", chatId, pdfPath);
        } catch (IOException e) {
            log.error("发送PDF生成事件失败，chatId: {}", chatId, e);
        }
    }

    /**
     * 获取当前活跃的chatId（如果只有一个连接）
     * 用于在工具中获取chatId
     */
    public String getCurrentChatId() {
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

}

