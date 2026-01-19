package com.ywy.ywyagent.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 资源下载服务 - 管理SseEmitter并发送资源下载事件
 */
@Service
public class ResourceService {

    private static final Logger log = LoggerFactory.getLogger(ResourceService.class);

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
     * 发送资源下载事件到前端
     *
     * @param chatId 聊天ID
     * @param resourcePath 资源文件的完整路径
     * @param fileName 文件名（用于显示）
     */
    public void sendResourceDownloadedEvent(String chatId, String resourcePath, String fileName) {
        SseEmitter emitter = activeEmitters.get(chatId);
        if (emitter == null) {
            log.warn("未找到对应的SSE连接，无法发送资源下载事件，chatId: {}", chatId);
            return;
        }

        try {
            // 转义路径中的反斜杠和引号，用于JSON格式
            String escapedPath = resourcePath.replace("\\", "\\\\").replace("\"", "\\\"");
            String escapedFileName = fileName != null ? fileName.replace("\"", "\\\"") : "";
            String jsonData = String.format("{\"resourcePath\":\"%s\",\"fileName\":\"%s\"}",
                    escapedPath, escapedFileName);

            emitter.send(SseEmitter.event()
                    .name("resourceDownloaded")
                    .data(jsonData));

            log.info("✅ 已发送资源下载事件到前端，chatId: {}, resourcePath: {}, fileName: {}",
                    chatId, resourcePath, fileName);
        } catch (IOException e) {
            log.error("发送资源下载事件失败，chatId: {}", chatId, e);
        }
    }

    /**
     * 获取当前活跃的chatId（如果只有一个连接）
     * 用于在工具中获取chatId
     */
    public String getCurrentChatId() {
        if (activeEmitters.size() == 1) {
            return activeEmitters.keySet().iterator().next();
        }
        return null;
    }
}

